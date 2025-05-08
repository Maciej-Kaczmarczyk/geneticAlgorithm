using System;

class NeuralNetwork
{
    static Random rand = new Random();

    const int inputSize = 2;
    const int hiddenSize = 2;
    const double learningRate = 0.1;
    const int epochs = 20000;

    static double[,] weightsInputHidden = new double[hiddenSize, inputSize + 1];
    static double[] weightsHiddenOutput = new double[hiddenSize + 1];

    static double Sigmoid(double x) => 1.0 / (1.0 + Math.Exp(-x));
    static double SigmoidDerivative(double x) => x * (1 - x);

    static void Main()
    {
        double[][] trainingData = new double[][]
        {
            new double[] {0, 0, 0},
            new double[] {0, 1, 1},
            new double[] {1, 0, 1},
            new double[] {1, 1, 0}
        };

        InitWeights();

        for (int epoch = 0; epoch < epochs; epoch++)
        {
            double totalError = 0.0;

            foreach (var data in trainingData)
            {
                double[] input = { data[0], data[1] };
                double expected = data[2];

                double[] hiddenOutputs = new double[hiddenSize];
                for (int i = 0; i < hiddenSize; i++)
                {
                    double sum = weightsInputHidden[i, inputSize];
                    for (int j = 0; j < inputSize; j++)
                        sum += weightsInputHidden[i, j] * input[j];
                    hiddenOutputs[i] = Sigmoid(sum);
                }

                double sumOutput = weightsHiddenOutput[hiddenSize];
                for (int i = 0; i < hiddenSize; i++)
                    sumOutput += weightsHiddenOutput[i] * hiddenOutputs[i];
                double output = Sigmoid(sumOutput);

                double error = expected - output;
                totalError += Math.Abs(error);

                double deltaOutput = error * SigmoidDerivative(output);
                double[] deltaHidden = new double[hiddenSize];
                for (int i = 0; i < hiddenSize; i++)
                    deltaHidden[i] = deltaOutput * weightsHiddenOutput[i] * SigmoidDerivative(hiddenOutputs[i]);


                for (int i = 0; i < hiddenSize; i++)
                    weightsHiddenOutput[i] += learningRate * deltaOutput * hiddenOutputs[i];
                weightsHiddenOutput[hiddenSize] += learningRate * deltaOutput * 1.0;

                for (int i = 0; i < hiddenSize; i++)
                {
                    for (int j = 0; j < inputSize; j++)
                        weightsInputHidden[i, j] += learningRate * deltaHidden[i] * input[j];
                    weightsInputHidden[i, inputSize] += learningRate * deltaHidden[i] * 1.0;
                }
            }

            if (epoch % 1000 == 0)
                Console.WriteLine($"Epoka {epoch}, błąd całkowity: {totalError:F4}");
        }

        Console.WriteLine("\nTest sieci:");
        foreach (var data in trainingData)
        {
            double[] input = { data[0], data[1] };
            double result = Predict(input);
            Console.WriteLine($"{input[0]} {input[1]} = {result:F4}");
        }
    }

    static void InitWeights()
    {
        for (int i = 0; i < hiddenSize; i++)
            for (int j = 0; j <= inputSize; j++)
                weightsInputHidden[i, j] = rand.NextDouble() * 2 - 1;

        for (int i = 0; i <= hiddenSize; i++)
            weightsHiddenOutput[i] = rand.NextDouble() * 2 - 1;
    }

    static double Predict(double[] input)
    {
        double[] hiddenOutputs = new double[hiddenSize];
        for (int i = 0; i < hiddenSize; i++)
        {
            double sum = weightsInputHidden[i, inputSize];
            for (int j = 0; j < inputSize; j++)
                sum += weightsInputHidden[i, j] * input[j];
            hiddenOutputs[i] = Sigmoid(sum);
        }

        double sumOutput = weightsHiddenOutput[hiddenSize];
        for (int i = 0; i < hiddenSize; i++)
            sumOutput += weightsHiddenOutput[i] * hiddenOutputs[i];
        return Sigmoid(sumOutput);
    }
}
