package main
import (
	"fmt"
	"./gonn"
)

func main(){
	nn := gonn.DefaultNetwork(2,30,1,true)
	inputs := [][]float64{
		[]float64{0,0},
		[]float64{0,1},
		[]float64{1,0},
		[]float64{1,1},
	}

	targets := [][]float64{
		[]float64{0},
		[]float64{1},
		[]float64{1},
		[]float64{0},
	}

	nn.Train(inputs,targets,1000)

	for _,p := range inputs{
		fmt.Println(nn.Forward(p))
	}

}