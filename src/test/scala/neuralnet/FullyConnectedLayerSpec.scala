package neuralnet

import chisel3._
import chisel3.tester._
import org.scalatest.FreeSpec
import chisel3.experimental.BundleLiterals._
import neuralnet.FullyConnectedLayer.NeuronState
import org.scalatest.FreeSpec

import FullyConnectedLayerSpec._

class FullyConnectedLayerSpec extends FreeSpec with ChiselScalatestTester {

  "FCLayer should initialize weights and bias" in {
    test(new FullyConnectedLayer) { dut =>
      dut.io.nextState.valid.poke(true.B)
      dut.io.nextState.bits.poke(NeuronState.init)
      dut.io.nextState.ready.expect(true.B)
      dut.clock.step(1)
      dut.io.weights.valid.poke(true.B)
      dut.io.weights.bits.foreach { i =>
        i.foreach(j => j.poke(1.F(TestDataWidth, TestBinaryPoint)))
      }
      dut.io.bias.bits.foreach(bit => bit.poke(0.5.F(TestDataWidth, TestBinaryPoint)))
      dut.io.bias.valid.poke(true.B)
      dut.io.weights.ready.expect(true.B)
      dut.io.bias.ready.expect(true.B)
      dut.clock.step(1)
    }
  }

  "FCLayer should calculate output values" in {
    test(new FullyConnectedLayer) { dut =>
      dut.io.nextState.valid.poke(true.B)
      dut.io.nextState.bits.poke(NeuronState.init)
      dut.clock.step(1)
      dut.io.weights.valid.poke(true.B)
      dut.io.bias.valid.poke(true.B)
      dut.io.weights.bits.foreach { i =>
        i.foreach(j => j.poke(2.F(TestDataWidth, TestBinaryPoint)))
      }
      dut.io.bias.bits.foreach(bit => bit.poke(0.5.F(TestDataWidth, TestBinaryPoint)))
      dut.clock.step(1)
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
}

object FullyConnectedLayerSpec {
  val TestInputNeurons = 2
  val TestOutputNeurons = 1
  val TestDataWidth = 32.W
  val TestBinaryPoint = 16.BP
}