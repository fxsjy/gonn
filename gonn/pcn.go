package gonn

import (
	"fmt"
	"math/rand"
	"time"
	"os"
	"encoding/json"
)

type PCNNetwork struct{
	InputCount int
	InputLayer []float64
	OutputLayer []float64
	WeightOutput [][]float64
	LastChangeOutput [][]float64
	Regression bool
	Rate1 float64
	Rate2 float64
}



func DumpPCN(fileName string, nn *PCNNetwork){
	out_f, err := os.OpenFile(fileName,os.O_CREATE,777)
	if err!=nil{
		panic("failed to dump the network to " + fileName)
	}
	defer out_f.Close()
	encoder := json.NewEncoder(out_f)
	encoder.Encode(nn)
}

func LoadPCN(fileName string) *PCNNetwork{
	in_f, err := os.Open(fileName)
	if err!=nil{
		panic("failed to load "+fileName)
	}
	defer in_f.Close()
	decoder := json.NewDecoder(in_f)
	nn := &PCNNetwork{}
	decoder.Decode(nn)
	//fmt.Println(nn)
	return nn
}



func DefaultPCNNetwork(iInputCount,iOutputCount int,iRegression bool) *PCNNetwork{
	return NewPCNNetwork(iInputCount,iOutputCount,iRegression,0.25,0.1)
}


func NewPCNNetwork(iInputCount,iOutputCount int,iRegression bool,iRate1,iRate2 float64) * PCNNetwork{
	self := &PCNNetwork{}
	self.InputCount = iInputCount
	rand.Seed(time.Now().UnixNano())
	self.InputLayer = make([]float64,iInputCount+1)
	self.OutputLayer = make([]float64,iOutputCount)
	self.WeightOutput = randomMatrix(iOutputCount, iInputCount+1, -1.0, 1.0)
	self.LastChangeOutput = makeMatrix(iOutputCount, iInputCount+1, 0.0)
	self.Regression = iRegression
	self.Rate1 = iRate1
	self.Rate2 = iRate2
	return self
}

func (self *PCNNetwork) Forward(input []float64) []float64{
	if len(input)+1 != len(self.InputLayer) {
		panic("amount of input variable doesn't match")
	}
	for i := 0; i < len(input); i++ {
		self.InputLayer[i] = input[i]
	}
	self.InputLayer[len(self.InputLayer)-1] = 1.0 //bias node for input layer

	for i := 0; i < len(self.OutputLayer); i++ {
		sum := 0.0
		for j := 0; j < len(self.InputLayer); j++ {
			sum += self.InputLayer[j] * self.WeightOutput[i][j]
		}
		if(self.Regression){
			self.OutputLayer[i] = sum
		}else{
			self.OutputLayer[i] = sigmoid(sum)
		}
	}

	return self.OutputLayer[:]
}


func (self *PCNNetwork) Feedback(target []float64) {
	for i := 0; i < len(self.OutputLayer); i++ {
		err_i := self.OutputLayer[i] - target[i]
		for j := 0; j < len(self.InputLayer); j++ {
			change := 0.0
			delta := 0.0
			if self.Regression {
				delta = err_i
			} else {
				delta = err_i * dsigmoid(self.OutputLayer[i])
			}
			change = self.Rate1*delta*self.InputLayer[j] + self.Rate2*self.LastChangeOutput[i][j]
			self.WeightOutput[i][j] -= change
			self.LastChangeOutput[i][j] = change
		}
	}
}


func (self *PCNNetwork) CalcError(target []float64) float64 {
	errSum := 0.0
	for i := 0; i < len(self.OutputLayer); i++ {
		err := self.OutputLayer[i] - target[i]
		errSum += 0.5 * err * err
	}
	return errSum
}

func (self *PCNNetwork) Train(inputs [][]float64, targets [][]float64, iteration int) {
	if len(inputs[0]) != self.InputCount {
		panic("amount of input variable doesn't match")
	}
	if len(targets[0]) != len(self.OutputLayer) {
		panic("amount of output variable doesn't match")
	}
	

	iter_flag := -1
	for i := 0; i < iteration; i++ {
		idx_ary := genRandomIdx(len(inputs))
		cur_err := 0.0
		for j := 0; j < len(inputs); j++ {
			self.Forward(inputs[idx_ary[j]])
			self.Feedback(targets[idx_ary[j]])
			cur_err += self.CalcError(targets[idx_ary[j]])
			if (j+1)%1000 == 0 {
				if iter_flag != i {
					fmt.Println("")
					iter_flag = i
				}
				fmt.Printf("iteration %v / progress %.2f %% \r", i+1, float64(j)*100/float64(len(inputs)))
			}
		}
		if (iteration >= 10 && (i+1)%(iteration/10) == 0) || iteration < 10 {
			fmt.Printf("\niteration %v err: %.5f", i+1, cur_err / float64(len(inputs)))
		}
	}
	fmt.Println("\ndone.")
}


func (self *PCNNetwork) TrainMap(inputs []map[int]float64, targets [][]float64, iteration int) {

	if len(targets[0]) != len(self.OutputLayer) {
		panic("amount of output variable doesn't match")
	}
	

	iter_flag := -1
	for i := 0; i < iteration; i++ {
		idx_ary := genRandomIdx(len(inputs))
		cur_err := 0.0
		for j := 0; j < len(inputs); j++ {
			self.ForwardMap(inputs[idx_ary[j]])
			self.FeedbackMap(targets[idx_ary[j]], inputs[idx_ary[j]] )
			cur_err += self.CalcError(targets[idx_ary[j]])
			if (j+1)%1000 == 0 {
				if iter_flag != i {
					fmt.Println("")
					iter_flag = i
				}
				fmt.Printf("iteration %v / progress %.2f %% \r", i+1, float64(j)*100/float64(len(inputs)))
			}
		}
		if (iteration >= 10 && (i+1)%(iteration/10) == 0) || iteration < 10 {
			fmt.Printf("\niteration %v err: %.5f", i+1, cur_err / float64(len(inputs)))
		}
	}
	fmt.Println("\ndone.")
}

func (self *PCNNetwork) ForwardMap(input map[int]float64) []float64{
	
	for i,_ := range input {
		self.InputLayer[i] = input[i]
	}
	self.InputLayer[len(self.InputLayer)-1] = 1.0 //bias node for input layer

	for i := 0; i < len(self.OutputLayer); i++ {
		sum := 0.0
		for j,_:= range input {
			sum += self.InputLayer[j] * self.WeightOutput[i][j]
		}
		if(self.Regression){
			self.OutputLayer[i] = sum
		}else{
			self.OutputLayer[i] = sigmoid(sum)
		}
	}

	return self.OutputLayer[:]
}


func (self *PCNNetwork) FeedbackMap(target []float64, input map[int]float64) {
	for i := 0; i < len(self.OutputLayer); i++ {
		err_i := self.OutputLayer[i] - target[i]
		for j,_:= range input {
			change := 0.0
			delta := 0.0
			if self.Regression {
				delta = err_i
			} else {
				delta = err_i * dsigmoid(self.OutputLayer[i])
			}
			change = self.Rate1*delta*self.InputLayer[j] + self.Rate2*self.LastChangeOutput[i][j]
			self.WeightOutput[i][j] -= change
			self.LastChangeOutput[i][j] = change
		}
	}
}

