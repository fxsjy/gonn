package main
import (
    "../gonn"
    "encoding/binary"
    "io"
    "os"
    "fmt"
    "flag"
)

const numLabels = 10
const epsilon = 0.001
const hiddenNodes = 100
const pixelRange = 255
const learningRate = 0.25
const momentum = 0.10

func ReadMNISTLabels (r io.Reader) (labels []byte) {
    header := [2]int32{}
    binary.Read(r, binary.BigEndian, &header)
    labels = make([]byte, header[1])
    r.Read(labels)
    return
}

func ReadMNISTImages (r io.Reader) (images [][]byte, width, height int) {
    header := [4]int32{}
    binary.Read(r, binary.BigEndian, &header)
    images = make([][]byte, header[1])
    width, height = int(header[2]), int(header[3])
    for i := 0; i < len(images); i++ {
        images[i] = make([]byte, width * height)
        r.Read(images[i])
    }
    return
}

func ImageString (buffer []byte, height, width int) (out string) {
    for i, y := 0, 0; y < height; y++ {
        for x := 0; x < width; x++ {
            if buffer[i] > 128 { out += "#" } else { out += " " }
            i++
        }
        out += "\n"
    }
    return
}

func OpenFile (path string) *os.File {
    file, err := os.Open(path)
    if (err != nil) {
        fmt.Println(err)
        os.Exit(-1)
    }
    return file
}

func pixelWeight (px byte) float64 {
   return float64(px) / pixelRange * 0.9 + 0.1 
}

func prepareX(M [][]byte) [][]float64{
    rows := len(M)
    result := make([][]float64,rows)
    for i:=0;i<rows;i++{
        result[i] = make([]float64,len(M[i]))
        for j:=0;j<len(M[i]);j++{
            result[i][j] = pixelWeight(M[i][j])
        }
    }
    return result
}

func prepareY(N []byte) [][]float64{
    result := make([][]float64,len(N))
    for i:=0;i<len(result);i++{
        tmp := make([]float64,10)
        for j:=0;j<10;j++{
            tmp[j] = 0.1
        }
        tmp[N[i]] = 0.9
        result[i] = tmp
    }
    return result
}

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

func main () {
    sourceLabelFile := flag.String("sl", "", "source label file")
    sourceImageFile := flag.String("si", "", "source image file")
    testLabelFile := flag.String("tl", "", "test label file")
    testImageFile := flag.String("ti", "", "test image file")
    
    flag.Parse()

    if *sourceLabelFile == "" || *sourceImageFile == "" {
        flag.Usage()
        os.Exit(-2)
    }

    fmt.Println("Loading training data...")
    labelData := ReadMNISTLabels(OpenFile(*sourceLabelFile))
    imageData, width, height := ReadMNISTImages(OpenFile(*sourceImageFile))

    fmt.Println(len(imageData),len(imageData[0]),width,height)
    fmt.Println(len(labelData),labelData[0:10])

    inputs := prepareX(imageData)
    targets := prepareY(labelData)

    //fmt.Println(imageData[0])
    //fmt.Println(inputs[:10])
    //fmt.Println(targets[:10])

    nn := gonn.NewNetwork(784,100,10,false,0.25,0.1)
    nn.Train(inputs,targets,10) //20 iterations

    var testLabelData []byte
    var testImageData [][]byte
    if *testLabelFile != "" && *testImageFile != "" {
        fmt.Println("Loading test data...")
        testLabelData = ReadMNISTLabels(OpenFile(*testLabelFile))
        testImageData, _, _ = ReadMNISTImages(OpenFile(*testImageFile))
    }

    test_inputs := prepareX(testImageData)
    test_targets := prepareY(testLabelData)

    //test_inputs = inputs[:1000]
    //test_targets = targets[:1000]
    correct_ct :=0
    for i,p := range(test_inputs){
        //fmt.Println(nn.Forward(p))
        y := argmax(nn.Forward(p))
        yy := argmax(test_targets[i])
        //fmt.Println(y,yy)
        if y == yy{
            correct_ct += 1
        }
    }

    fmt.Println("correct rate: ", float64(correct_ct)/ float64(len(test_inputs)), correct_ct,len(test_inputs))
}
