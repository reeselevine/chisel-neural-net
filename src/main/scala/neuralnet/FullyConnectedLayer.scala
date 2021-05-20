package neuralnet

import chisel3._
import chisel3.experimental.{ChiselEnum, FixedPoint}
import chisel3.util._
import neuralnet.FullyConnectedLayer._

object FullyConnectedLayer {
  val InputSize = 2
  val OutputSize = 1
  val DataWidth = 8.W
  val DataBinaryPoint = 4.BP
  object NeuronState extends ChiselEnum {
    val ready, init, forwardProp, backwardProp = Value
  }
}

class FullyConnectedLayerIO extends Bundle {
  val weights = Flipped(Decoupled(Vec(InputSize, Vec(OutputSize, FixedPoint(DataWidth, DataBinaryPoint)))))
  val bias = Flipped(Decoupled(Vec(OutputSize, FixedPoint(DataWidth, DataBinaryPoint))))
  val input = Flipped(Decoupled(Vec(InputSize, FixedPoint(DataWidth, DataBinaryPoint))))
  val output = Decoupled(Vec(OutputSize, FixedPoint(DataWidth, DataBinaryPoint)))
  val nextState = Flipped(Decoupled(NeuronState()))
}

class FullyConnectedLayer extends Module {
  val io = IO(new FullyConnectedLayerIO)
  val state = RegInit(NeuronState.ready)
  val weights = VecInit(Seq.tabulate(InputSize)(VecInit(Seq.fill(OutputSize)(0.F(DataWidth, DataBinaryPoint)))))
  val bias = VecInit(Seq.fill(OutputSize)(0.F(DataWidth, DataBinaryPoint)))
  switch(state) {
    is(NeuronState.ready) {
      when(io.nextState.fire) {
        state := io.nextState.bits
      }
    }
    is(NeuronState.init) {
      weights := io.weights.bits
      bias := io.bias.bits
      state := NeuronState.ready
    }
  }
}
