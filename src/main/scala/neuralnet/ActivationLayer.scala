package neuralnet

import chisel3._
import chisel3.experimental.FixedPoint
import chisel3.util._
import neuralnet.NeuralNet.{DataBinaryPoint, DataWidth, NeuronState}

case class ActivationLayerParams(size: Int)

class ActivationLayerIO(params: ActivationLayerParams) extends Bundle {
  val input = Flipped(Decoupled(Vec(params.size, FixedPoint(DataWidth, DataBinaryPoint))))
  val output = Decoupled(Vec(params.size, FixedPoint(DataWidth, DataBinaryPoint)))
  val nextState = Flipped(Decoupled(NeuronState()))
}

class ActivationLayer(params: ActivationLayerParams) extends Module {

  val io = IO(new ActivationLayerIO(params))
  val state = RegInit(NeuronState.ready)

  io.input.ready := true.B
  io.nextState.ready := true.B
  io.output.valid := false.B
  io.output.bits := VecInit(Seq.fill(params.size)(0.F(DataWidth, DataBinaryPoint)))

}
