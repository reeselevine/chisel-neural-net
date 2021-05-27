package neuralnet

import chisel3._
import chisel3.tester._
import org.scalatest.FreeSpec
import FullyConnectedLayerSpec._
import chisel3.experimental.FixedPoint
import neuralnet.NeuralNet.{DataBinaryPoint, DataWidth, NeuronState}

class FullyConnectedLayerSpec extends FreeSpec with ChiselScalatestTester {

  "FCLayer should calculate output values" in {
    test(new TestFullyConnectedLayer(inputWeightValue = 2, inputBiasValue = 0.5)) { dut =>
      dut.io.nextState.valid.poke(true.B)
      dut.io.nextState.ready.expect(true.B)
      dut.io.nextState.bits.poke(NeuronState.forwardProp)
      dut.clock.step(1)
      dut.io.input.valid.poke(true.B)
      dut.io.input.ready.expect(true.B)
      dut.io.input.bits.foreach(bit => bit.poke(1.F(DataWidth, DataBinaryPoint)))
      dut.io.output.valid.expect(true.B)
      dut.io.output.bits.foreach(bit => bit.expect(4.5.F(DataWidth, DataBinaryPoint)))
    }
  }

  class TestFullyConnectedLayer(
                                 inputWeightValue: Double,
                                 inputBiasValue: Double,
                                 params: FullyConnectedLayerParams = FullyConnectedLayerParams(TestInputNeurons, TestOutputNeurons, TestAdjust))
    extends FullyConnectedLayer(params) {
    override def getInitialWeights(): Vec[Vec[FixedPoint]] = {
      VecInit(Seq.fill(params.inputSize)(VecInit(Seq.fill(params.outputSize)(inputWeightValue.F(DataWidth, DataBinaryPoint)))))
    }

    override def getInitialBias(): Vec[FixedPoint] = {
      VecInit(Seq.fill(params.outputSize)(inputBiasValue.F(DataWidth, DataBinaryPoint)))
    }
  }
}

object FullyConnectedLayerSpec {
  val TestInputNeurons = 2
  val TestOutputNeurons = 1
  val TestAdjust = 0.5
}