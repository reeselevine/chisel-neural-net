package neuralnet

import chisel3._
import chisel3.experimental.FixedPoint
import chisel3.util._
import neuralnet.NeuralNet.{DataBinaryPoint, DataWidth, NeuronState}

/** Implements an activation layer, for creating non-linear neural nets */
class ActivationLayer(params: LayerParams) extends Layer(params) {

  val state = RegInit(NeuronState.ready)

  io.input.ready := true.B
  io.nextState.ready := true.B
  io.output.valid := false.B
  io.output.bits := VecInit(Seq.fill(params.outputSize)(0.F(DataWidth, DataBinaryPoint)))

  switch(state) {
    is(NeuronState.ready) {
      when(io.nextState.fire()) {
        state := io.nextState.bits
      }
    }
    // By default this uses Relu activation. We may want to parameterize this later.
    is(NeuronState.forwardProp) {
      when(io.input.fire()) {
        val inputData = io.input.bits
        (0 until params.outputSize).foreach { i =>
          when(inputData(i) > 0.F(DataWidth, DataBinaryPoint)) {
            io.output.bits(i) := inputData(i)
          } .otherwise {
            io.output.bits(i) := 0.F(DataWidth, DataBinaryPoint)
          }
        }
        io.output.valid := true.B
        state := NeuronState.ready
      }
    }
  }
}
