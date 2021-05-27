package neuralnet

import chisel3._
import chisel3.experimental.{ChiselEnum, FixedPoint}
import chisel3.util._
import neuralnet.FullyConnectedLayer._

import scala.util.Random

object FullyConnectedLayer {
  val InputSize = 2
  val OutputSize = 1
  val DataWidth = 32.W
  val DataBinaryPoint = 16.BP
  val Adjust = 0.5
  object NeuronState extends ChiselEnum {
    val ready, reset, forwardProp, backwardProp = Value
  }
}

class FullyConnectedLayerIO extends Bundle {
  val input = Flipped(Decoupled(Vec(InputSize, FixedPoint(DataWidth, DataBinaryPoint))))
  val output = Decoupled(Vec(OutputSize, FixedPoint(DataWidth, DataBinaryPoint)))
  val nextState = Flipped(Decoupled(NeuronState()))
}

/**
 * Implements a fully connected layer in a neural net, where each input neuron affects each output neuron
 * with weights defined in a stored matrix.
 */
class FullyConnectedLayer extends Module {
  val r = Random
  val io = IO(new FullyConnectedLayerIO)
  val state = RegInit(NeuronState.ready)
  val weightsValues = getInitialWeights()
  val weights = RegInit(weightsValues)
  val biasValues = getInitialBias()
  val bias = RegInit(biasValues)

  io.input.ready := true.B
  io.nextState.ready := true.B
  io.output.bits := VecInit(Seq.fill(OutputSize)(0.F(DataWidth, DataBinaryPoint)))

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
        (0 until OutputSize).foreach { j =>
          val dotProduct = (0 until InputSize).foldLeft(0.F(DataWidth, DataBinaryPoint)) { (sum, i) =>
            sum + inputData(i) * weights(i)(j)
          }
          io.output.bits(j) := dotProduct + bias(j)
        }
        io.output.valid := true.B
      }
    }
  }

  def getInitialWeights(): Vec[Vec[FixedPoint]] = {
    VecInit(Seq.fill(InputSize)(VecInit(Seq.tabulate(OutputSize)(_ =>
      (r.nextDouble() - Adjust).F(DataWidth, DataBinaryPoint)))))
  }

  def getInitialBias(): Vec[FixedPoint] = {
    VecInit(Seq.tabulate(OutputSize)(_ =>
      (r.nextDouble() - Adjust).F(DataWidth, DataBinaryPoint)))
  }
}
