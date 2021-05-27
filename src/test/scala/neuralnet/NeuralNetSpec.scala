package neuralnet

import chisel3._
import chisel3.tester._
import org.scalatest.FreeSpec
import FullyConnectedLayerSpec._
import chisel3.experimental.FixedPoint
import neuralnet.NeuralNet.{DataBinaryPoint, DataWidth, FCLayer, NeuronState}
import neuralnet.NeuralNetSpec.buildBasicNet

class NeuralNetSpec extends FreeSpec with ChiselScalatestTester {

  "A neural net should initialize" in {
    test(buildBasicNet()) { dut =>
      dut.io.numSamples.ready.expect(true.B)
    }
  }
}

object NeuralNetSpec {

  def buildBasicNet(): NeuralNet = {
    val params = NeuralNetParams(inputSize = 2, outputSize = 1, layers = Seq(FCLayer(fcLayerDefaultParams)))
    new NeuralNet(params)
  }
}
