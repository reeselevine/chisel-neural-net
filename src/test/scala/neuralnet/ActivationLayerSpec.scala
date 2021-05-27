package neuralnet

import chisel3._
import chisel3.tester._
import org.scalatest.FreeSpec
import neuralnet.ActivationLayerSpec._
import neuralnet.NeuralNet.{DataBinaryPoint, DataWidth, NeuronState}

class ActivationLayerSpec extends FreeSpec with ChiselScalatestTester {

  "Activation layer should activate values" in {
    test(new ActivationLayer(defaultParams)) { dut =>
      dut.io.nextState.valid.poke(true.B)
      dut.io.nextState.ready.expect(true.B)
      dut.io.nextState.bits.poke(NeuronState.forwardProp)
      dut.clock.step(1)
      dut.io.input.valid.poke(true.B)
      dut.io.input.ready.expect(true.B)
      dut.io.input.bits.zipWithIndex.map {
        case (bit, i) => bit.poke(defaultInput(i).F(DataWidth, DataBinaryPoint))
      }
      dut.io.output.valid.expect(true.B)
      dut.io.output.bits.zipWithIndex.map {
        case (bit, i) => bit.expect(reluOutput(i).F(DataWidth, DataBinaryPoint))
      }
    }
  }
}

object ActivationLayerSpec {
  val defaultParams = ActivationLayerParams(4)
  val defaultInput = Seq(1.0, -1.0, 0.5, 0.0)
  val reluOutput =  Seq(1.0, 0.0, 0.5, 0.0)
}
