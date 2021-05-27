package neuralnet

import chisel3._
import chisel3.tester._
import org.scalatest.FreeSpec
import FullyConnectedLayerSpec._
import chisel3.experimental.FixedPoint
import neuralnet.NeuralNet.{DataBinaryPoint, DataWidth, FCLayer, NeuronState}
import neuralnet.NeuralNetSpec.buildBasicNet

class NeuralNetSpec extends FreeSpec with ChiselScalatestTester {

  "A neural net should initialize and predict multiple samples" in {
    test(buildBasicNet()) { dut =>
      dut.io.numSamples.ready.expect(true.B)
      dut.io.numSamples.valid.poke(true.B)
      dut.io.numSamples.bits.poke(2.U)
      dut.io.sample.foreach(bit => bit.poke(1.F(DataWidth, DataBinaryPoint)))
      dut.io.predict.poke(true.B)
      dut.clock.step(1)
      dut.io.result.ready.poke(true.B)
      dut.clock.step(1)
      dut.io.result.valid.expect(true.B)
      dut.clock.step(1)
      dut.io.result.valid.expect(false.B)
      dut.clock.step(1)
      dut.io.result.valid.expect(true.B)
    }
  }
}

object NeuralNetSpec {

  def buildBasicNet(): NeuralNet = {
    val params = NeuralNetParams(inputSize = 2, outputSize = 1, layers = Seq(FCLayer(fcLayerDefaultParams)))
    new NeuralNet(params)
  }
}
