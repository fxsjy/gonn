package main

import (
	"fmt"
	"./gonn"
	"math"
	"os"
)

func main(){
	out_f,_ := os.OpenFile("sin.out",os.O_CREATE,777)
	defer out_f.Close()

	nn := gonn.DefaultNetwork(1,10,1,true)
	train_inputs := make([][]float64,100)
	train_targets := make([][]float64,100)

	for i:=0;i<len(train_inputs);i++{
		train_inputs[i] = []float64{ float64(i) / 20.0 }
		train_targets[i] = []float64{  math.Sin(train_inputs[i][0] ) }
	}

	nn.Train(train_inputs,train_targets,1000)

	for i:=0;i<len(train_inputs);i++{
		x := []float64{ float64(i) / 23.0 }
		fmt.Fprintln(out_f,x[0], nn.Forward(x)[0])
	}

}

