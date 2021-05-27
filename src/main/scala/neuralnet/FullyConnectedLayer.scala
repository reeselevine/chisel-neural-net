package neuralnet

import chisel3._
import chisel3.experimental.FixedPoint
import chisel3.util._
import neuralnet.NeuralNet.{DataBinaryPoint, DataWidth, NeuronState}

import scala.util.Random

case class FullyConnectedLayerParams(inputSize: Int, outputSize: Int, adjust: Double)

class FullyConnectedLayerIO(params: FullyConnectedLayerParams) extends Bundle {
  val input = Flipped(Decoupled(Vec(params.inputSize, FixedPoint(DataWidth, DataBinaryPoint))))
  val output = Decoupled(Vec(params.outputSize, FixedPoint(DataWidth, DataBinaryPoint)))
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
  val weightsValues = getInitialWeights()
  val weights = RegInit(weightsValues)
  val biasValues = getInitialBias()
  val bias = RegInit(biasValues)

  io.input.ready := true.B
  io.nextState.ready := true.B
  io.output.bits := VecInit(Seq.fill(params.outputSize)(0.F(DataWidth, DataBinaryPoint)))

  io.output.valid := false.B

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
          io.output.bits(j) := dotProduct + bias(j)
        }
        io.output.valid := true.B
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
