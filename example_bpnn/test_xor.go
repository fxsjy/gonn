package main
import (
	"fmt"
	"../gonn"
)


func main(){
	nn := gonn.DefaultNetwork(2,2,1,true)
	inputs := [][]float64{
		[]float64{0,0},
		[]float64{0,1},
		[]float64{1,0},
		[]float64{1,1},
	}

	targets := [][]float64{
		[]float64{0}, //0 xor 0 == 0
		[]float64{1}, //0 xor 1 == 1
		[]float64{1}, //1 xor 0 == 1
		[]float64{0}, //1 xor 1 == 0
	}

	nn.Train(inputs,targets,1000)

	for _,p := range inputs{
		fmt.Printf("%.1f\n",nn.Forward(p)[0])
	}

}