package neuralnet

import chisel3._
import chisel3.tester._
import org.scalatest.FreeSpec
import chisel3.experimental.BundleLiterals._
import neuralnet.FullyConnectedLayer._
import org.scalatest.FreeSpec
import FullyConnectedLayerSpec._
import chisel3.experimental.FixedPoint

class FullyConnectedLayerSpec extends FreeSpec with ChiselScalatestTester {

  "FCLayer should calculate output values" in {
    test(new TestFullyConnectedLayer(inputWeightValue = 2, inputBiasValue = 0.5)) { dut =>
      dut.io.nextState.valid.poke(true.B)
      dut.io.nextState.bits.poke(NeuronState.forwardProp)
      dut.clock.step(1)
      dut.io.input.valid.poke(true.B)
      dut.io.input.bits.foreach(bit => bit.poke(1.F(TestDataWidth, TestBinaryPoint)))
      dut.io.output.valid.expect(true.B)
      dut.io.output.bits.foreach(bit => bit.expect(4.5.F(TestDataWidth, TestBinaryPoint)))
      dut.clock.step(1)
    }
  }

  class TestFullyConnectedLayer(
                                 inputWeightValue: Double,
                                 inputBiasValue: Double) extends FullyConnectedLayer {
    override def getInitialWeights(): Vec[Vec[FixedPoint]] = {
      VecInit(Seq.fill(InputSize)(VecInit(Seq.fill(OutputSize)(inputWeightValue.F(DataWidth, DataBinaryPoint)))))
    }

    override def getInitialBias(): Vec[FixedPoint] = {
      VecInit(Seq.fill(OutputSize)(inputBiasValue.F(DataWidth, DataBinaryPoint)))
    }
  }
}

object FullyConnectedLayerSpec {
  val TestInputNeurons = 2
  val TestOutputNeurons = 1
  val TestDataWidth = 32.W
  val TestBinaryPoint = 16.BP
}