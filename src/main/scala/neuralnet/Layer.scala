package neuralnet

import chisel3._
import chisel3.experimental.FixedPoint
import chisel3.util._
import neuralnet.NeuralNet.{DataBinaryPoint, DataWidth, NeuronState}

class LayerIO(params: LayerParams) extends Bundle {
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

class LayerParams(val inputSize: Int, val outputSize: Int)

object LayerParams {

  def apply(inputSize: Int, outputSize: Int) = new LayerParams(inputSize, outputSize)
}

abstract class Layer(val params: LayerParams) extends Module {

  val io = IO(new LayerIO(params))
}
