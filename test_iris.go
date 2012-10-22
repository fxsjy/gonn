package main

import (
	"fmt"
	"io/ioutil"
	"strings"
	"strconv"
	"os"
	//"math"
	"./gonn"
)

func argmax(A []float64) int{
	x := 0 
	v := -1.0
	for i,a := range(A){
		if a>v{
			x = i
			v = a
		}
	}
	return x
}


func main(){
	f,_ := os.Open("iris_proc.data")
	content,_ := ioutil.ReadAll(f)
	s_content := string(content)
	lines := strings.Split(s_content,"\n")
	nn := gonn.DefaultNetwork(4,10,3,false)
	nn.Start()

	inputs := make([][]float64,0)
	targets := make([][]float64,0)
	for _,line := range lines{
		if len(line)==0{
			continue
		}
		tup := strings.Split(line,",")
		pattern := tup[:len(tup)-1]
		target := tup[len(tup)-1]
		X := make([]float64,0)
		for _,x := range pattern{
			f_x,_:= strconv.ParseFloat(x,64)
			X = append(X,f_x)
		}
		inputs = append(inputs,X)
		Y := make([]float64,3)
		i_target,_ := strconv.Atoi(target)
		Y[i_target] = 1.0
		targets = append(targets,Y)
	}
	train_inputs := make([][]float64,0)
	train_targets := make([][]float64,0)

	test_inputs := make([][]float64,0)
	test_targets := make([][]float64,0)

	for i,x := range inputs{
		if i%3==0{
			test_inputs = append(test_inputs, x)
		}else{
			train_inputs = append(train_inputs, x)
		}
	}

	for i,y := range targets{
		if i%3==0{
			test_targets = append(test_targets,y)
		}else{
			train_targets = append(train_targets,y)
		}
	}

	//fmt.Println(train_inputs,train_targets)
	nn.Train(train_inputs,train_targets,1000)
	err_count := 0.0
	for i:=0;i<len(test_inputs);i++{
		output := nn.Forward(test_inputs[i])
		calc := argmax(output)
		expect := argmax(test_targets[i])
		fmt.Println(calc,expect)
		if calc!=expect{
			err_count += 1
		}
	}
	fmt.Println("success rate:",1.0 - err_count/float64(len(test_inputs)))

	nn.Stop()
}
