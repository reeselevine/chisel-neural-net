package neuralnet

import chisel3._
import chisel3.experimental.FixedPoint
import chisel3.util._
import neuralnet.NeuralNet.{DataBinaryPoint, DataWidth, NeuronState}

import scala.util.Random

case class FullyConnectedLayerParams(inputSize: Int, outputSize: Int, adjust: Double)

class FullyConnectedLayerIO(params: FullyConnectedLayerParams) extends Bundle {
  // Input from left layer.
  val input = Flipped(Decoupled(Vec(params.inputSize, FixedPoint(DataWidth, DataBinaryPoint))))
  // Error passed in by the right layer (this layer's current output).
  val output_error = Flipped(Decoupled(Vec(params.outputSize, FixedPoint(DataWidth, DataBinaryPoint))))
  // Error passed to the left layer (this current layer's input).
  val input_error = Decoupled(Vec(params.inputSize, FixedPoint(DataWidth, DataBinaryPoint)))
  // Output to right layer.
  val output = Decoupled(Vec(params.outputSize, FixedPoint(DataWidth, DataBinaryPoint)))
  // Next state command from NeuralNet.
  val nextState = Flipped(Decoupled(NeuronState()))
}

/**
 * Implements a fully connected layer in a neural net, where each input neuron affects each output neuron
 * with weights defined in a stored matrix.
 */
class FullyConnectedLayer(params: FullyConnectedLayerParams) extends Module {
  val r = Random
  val io = IO(new FullyConnectedLayerIO(params))
  val state = RegInit(NeuronState.ready)
  val weights = RegInit(getInitialWeights()) // Input weights.
  val bias = RegInit(getInitialBias())       // Neuron biases.

  io.input.ready := true.B
  io.nextState.ready := true.B
  io.output.bits := VecInit(Seq.fill(params.outputSize)(0.F(DataWidth, DataBinaryPoint)))

  io.output.valid := false.B

  io.output_error.ready := true.B
  io.input_error.valid := false.B
  io.input_error.bits := VecInit(Seq.fill(params.inputSize)(0.F(DataWidth, DataBinaryPoint)))

  switch(state) {
    // Intermediate state, used to transition between initialization, training, and predicting.
    is(NeuronState.ready) {
      when(io.nextState.fire) {
        state := io.nextState.bits
      }
    }
    // Initializes (or resets) this layer with the given weights and bias.
    is(NeuronState.reset) {
      weights := weightsValues
      bias := biasValues
      state := NeuronState.ready
    }
    // Performs forward propagation, for either training or prediction.
    is(NeuronState.forwardProp) {
      when(io.input.fire()) {
        val inputData = io.input.bits
        // computes the dot product for each output neuron using the input neurons and their defined weights.
        (0 until params.outputSize).foreach { j =>
          val dotProduct = (0 until params.inputSize).foldLeft(0.F(DataWidth, DataBinaryPoint)) { (sum, i) =>
            sum + inputData(i) * weights(i)(j)
          }
          
          // ReLu activation. 
          val net = dotProduct + bias(j)
          when (net > 0.F(DataWidth, DataBinaryPoint)) {
            io.output.bits(j) := net
          } .otherwise {
            io.output.bits(j) := 0.F(DataWidth, DataBinaryPoint)
          }
        }
        io.output.valid := true.B

        // Wait for next state change.
        state := NeuronState.ready
      }
    }
    // Perform back propagation.
    // https://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/
    is(NeuronState.backwardProp) {
      when (io.output_error.fire()) {
        // Compute own deltas (gradient).
        val deltas = (0 until params.outputSize).map { j =>
          // ReLu derivative.
          // TODO: necessary use of var?
          var deriv = 0.F(DataWidth, DataBinaryPoint);
          when (io.output.bits(j) > 0.F(DataWidth, DataBinaryPoint)) {
            deriv = 1.F(DataWidth, DataBinaryPoint)
          }

          io.output_error(j) * deriv;
        }
      
        // Update weights and biases.
        for (j <- 0 until params.outputSize) {
          for (i <- 0 until params.inputSize) {
            weights(i)(j) := weights(i)(j) + LearningRate * deltas(j) * io.input(i)
          }

          bias(j) := bias(j) + LearningRate * deltas(j)
        }

        // Compute error to pass to left layer.
        for (i <- 0 until params.inputSize) {
          val dotPdt = (0 until params.outputSize)
            .foldLeft(0.F(DataWidth, DataBinaryPoint)) { (sum, j) =>
              sum + weights(i)(j) * delta(j)
          }

          io.input_error.bits(i) := dotPdt
        }
        io.input_error.valid := true.B

        // Wait for next state change.
        state := NeuronState.ready
      }
    }
  }

  def getInitialWeights(): Vec[Vec[FixedPoint]] = {
    VecInit(Seq.fill(params.inputSize)(VecInit(Seq.tabulate(params.outputSize)(_ =>
      (r.nextDouble() - params.adjust).F(DataWidth, DataBinaryPoint)))))
  }

  def getInitialBias(): Vec[FixedPoint] = {
    VecInit(Seq.tabulate(params.outputSize)(_ =>
      (r.nextDouble() - params.adjust).F(DataWidth, DataBinaryPoint)))
  }
}
