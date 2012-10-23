package gonn

import (
	"math"
	"math/rand"
	"fmt"
)

type NeuralNetwork struct{
	mHiddenLayer []float64
	mInputLayer  []float64
	mOutputLayer []float64
	mWeightHidden [][]float64
	mWeightOutput [][]float64
	mErrOutput []float64
	mErrHidden []float64
	mLastChangeHidden [][]float64
	mLastChangeOutput [][]float64
	mRegression bool
	mRate1 float64 //learning rate
	mRate2 float64
}

func sigmoid(X float64) float64{
	return math.Tanh(X)
}

func dsigmoid(Y float64) float64{
	return 1-Y*Y
}


func makeMatrix(rows,colums int, value float64) [][]float64{
	mat := make([][]float64,rows)
	for i:=0;i<rows;i++{
		mat[i] = make([]float64,colums)
		for j:=0;j<colums;j++{
			mat[i][j] = value
		}
	}
	return mat
}

func randomMatrix(rows,colums int, lower, upper float64) [][]float64{
	mat := make([][]float64,rows)
	for i:=0;i<rows;i++{
		mat[i] = make([]float64,colums)
		for j:=0;j<colums;j++{
			mat[i][j] = rand.Float64()*(upper-lower) + lower
		}
	}
	return mat
}

func DefaultNetwork(iInputCount,iHiddenCount,iOutputCount int, iRegression bool) (*NeuralNetwork) {
	return NewNetwork(iInputCount,iHiddenCount,iOutputCount,iRegression, 0.01,0.001)
}


func NewNetwork(iInputCount,iHiddenCount,iOutputCount int, iRegression bool,iRate1,iRate2 float64) (*NeuralNetwork){
	iInputCount +=1
	network := &NeuralNetwork{}
	network.mRegression = iRegression
	network.mRate1 = iRate1
	network.mRate2 = iRate2
	network.mInputLayer = make([]float64,iInputCount)
	network.mHiddenLayer = make([]float64,iHiddenCount)
	network.mOutputLayer = make([]float64,iOutputCount)
	network.mErrOutput = make([]float64,iOutputCount)
	network.mErrHidden = make([]float64,iHiddenCount)

	network.mWeightHidden = randomMatrix(iInputCount,iHiddenCount,-0.2,0.2)
	network.mWeightOutput = randomMatrix(iHiddenCount,iOutputCount,-2.0,2.0)

	network.mLastChangeHidden = makeMatrix(iInputCount,iHiddenCount,0.0)
	network.mLastChangeOutput = makeMatrix(iHiddenCount,iOutputCount,0.0)

	return network
}


func (self * NeuralNetwork) Forward(input []float64 ) (output []float64){
	if len(input)+1 != len(self.mInputLayer){
		panic("amount of input variable doesn't match")
	}
	for i:=0;i<len(input);i++{
		self.mInputLayer[i] = input[i]
	}
	self.mInputLayer[len(self.mInputLayer)-1] = 1.0 //bias node
	
	for i:=0;i<len(self.mHiddenLayer);i++{
			sum := 0.0
			for j:=0;j<len(self.mInputLayer);j++{
				sum += self.mInputLayer[j] * self.mWeightHidden[j][i]
			}
			self.mHiddenLayer[i] = sigmoid(sum)
	}

	for i:=0;i<len(self.mOutputLayer);i++{
		sum := 0.0
		for j:=0; j<len(self.mHiddenLayer);j++{
			sum += self.mHiddenLayer[j] * self.mWeightOutput[j][i]
		}
		if(self.mRegression){
			self.mOutputLayer[i] = sum
		}else{
			self.mOutputLayer[i] = sigmoid(sum)
		}
	}
	return self.mOutputLayer[:]
}

func (self * NeuralNetwork) Feedback(target []float64) {
	for i:=0;i<len(self.mOutputLayer);i++{
		self.mErrOutput[i] = self.mOutputLayer[i] - target[i]
	}

	for i:=0;i<len(self.mHiddenLayer);i++{
		err := 0.0
		for j:=0;j<len(self.mOutputLayer);j++{
			err += self.mErrOutput[j] * self.mWeightOutput[i][j]
		}
		self.mErrHidden[i] = err
	}

	for i:=0;i<len(self.mOutputLayer);i++{
		for j:=0;j<len(self.mHiddenLayer);j++{
			change := 0.0
			if(self.mRegression){
				change = self.mErrOutput[i] * self.mHiddenLayer[j]
			}else{
				change = self.mErrOutput[i] * self.mHiddenLayer[j] * dsigmoid(self.mOutputLayer[i])
			}
			self.mWeightOutput[j][i] -= (self.mRate1*change + self.mRate2* self.mLastChangeOutput[j][i])
			self.mLastChangeOutput[j][i] = change 
		}
	}

	for i:=0;i<len(self.mHiddenLayer);i++{
		for j:=0;j<len(self.mInputLayer);j++{
			change := self.mErrHidden[i] * self.mInputLayer[j] * dsigmoid(self.mHiddenLayer[i])
			self.mWeightHidden[j][i] -= (self.mRate1*change + self.mRate2*self.mLastChangeHidden[j][i])
			self.mLastChangeHidden[j][i] = change
		}
	}
}


func (self * NeuralNetwork) CalcError( target []float64) float64{
	errSum := 0.0
	for i:=0;i<len(self.mOutputLayer);i++{
		err := self.mOutputLayer[i] - target[i]
		errSum += 0.5 * err * err
	}
	return errSum
}

func genRandomIdx(N int) []int{
	A := make([]int,N)
	for i:=0;i<N;i++{
		A[i]=i
	}
	//randomize
	for i:=0;i<N;i++{
		j := i+int(rand.Float64() * float64 (N-i))
		A[i],A[j] = A[j],A[i]
	}
	return A
}

func (self * NeuralNetwork) Train(inputs [][]float64, targets [][]float64, iteration int) {
	if len(inputs[0])+1 != len(self.mInputLayer){
		panic("amount of input variable doesn't match")
	}
	if len(targets[0]) != len(self.mOutputLayer){
		panic("amount of output variable doesn't match")
	}
	old_err1 := 1.0
	old_err2 := 2.0
	
	for i:=0;i<iteration;i++{
		idx_ary := genRandomIdx(len(inputs))
		for j:=0;j<len(inputs);j++{
			self.Forward(inputs[idx_ary[j]])
			self.Feedback(targets[idx_ary[j]])
		}
		if i%100==0 {
			last_target := targets[len(targets)-1]
			cur_err := self.CalcError(last_target)
			fmt.Println("err: ", cur_err)
			if (old_err2 - old_err1 < 0.001) && (old_err1 - cur_err  < 0.001){//early stop
				break
			}	
			old_err2 = old_err1
			old_err1 = cur_err
		}
	}
}
