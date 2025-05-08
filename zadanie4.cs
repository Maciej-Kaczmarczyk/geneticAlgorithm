using System;

public class Neuron
{
    public double[] Weights;
    public double Bias;
    public double Output;
    public double Delta;

    public Neuron(int inputCount, Random rand)
    {
        Weights = new double[inputCount];
        for (int i = 0; i < inputCount; i++)
            Weights[i] = rand.NextDouble() * 2 - 1;

        Bias = rand.NextDouble() * 2 - 1;
    }

    public double Activate(double[] inputs)
    {
        double sum = Bias;
        for (int i = 0; i < inputs.Length; i++)
            sum += inputs[i] * Weights[i];

        Output = 1.0 / (1.0 + Math.Exp(-sum));
        return Output;
    }

    public double SigmoidDerivative()
    {
        return Output * (1 - Output);
    }
}

public class Layer
{
    public Neuron[] Neurons;

    public Layer(int neuronCount, int inputCount, Random rand)
    {
        Neurons = new Neuron[neuronCount];
        for (int i = 0; i < neuronCount; i++)
            Neurons[i] = new Neuron(inputCount, rand);
    }

    public double[] FeedForward(double[] inputs)
    {
        double[] outputs = new double[Neurons.Length];
        for (int i = 0; i < Neurons.Length; i++)
            outputs[i] = Neurons[i].Activate(inputs);
        return outputs;
    }
}

public class NeuralNetwork
{
    private Layer Hidden;
    private Neuron OutputNeuron;
    private double LearningRate = 0.1;

    public NeuralNetwork(Random rand)
    {
        Hidden = new Layer(2, 2, rand);
        OutputNeuron = new Neuron(2, rand);
    }

    public double FeedForward(double[] inputs)
    {
        double[] hiddenOutputs = Hidden.FeedForward(inputs);
        return OutputNeuron.Activate(hiddenOutputs);
    }

    public void Train(double[] inputs, double target)
    {
        double output = FeedForward(inputs);

        double error = target - output;
        OutputNeuron.Delta = error * OutputNeuron.SigmoidDerivative();

        for (int i = 0; i < Hidden.Neurons.Length; i++)
        {
            Neuron h = Hidden.Neurons[i];
            h.Delta = OutputNeuron.Delta * OutputNeuron.Weights[i] * h.SigmoidDerivative();
        }

        for (int i = 0; i < OutputNeuron.Weights.Length; i++)
            OutputNeuron.Weights[i] += LearningRate * OutputNeuron.Delta * Hidden.Neurons[i].Output;

        OutputNeuron.Bias += LearningRate * OutputNeuron.Delta;

        for (int i = 0; i < Hidden.Neurons.Length; i++)
        {
            for (int j = 0; j < Hidden.Neurons[i].Weights.Length; j++)
                Hidden.Neurons[i].Weights[j] += LearningRate * Hidden.Neurons[i].Delta * inputs[j];

            Hidden.Neurons[i].Bias += LearningRate * Hidden.Neurons[i].Delta;
        }

    }
}

public class Program
{
    public static void Main()
    {
        var rand = new Random();
        var nn = new NeuralNetwork(rand);

        double[][] inputs = {
            new double[] { 0, 0 },
            new double[] { 0, 1 },
            new double[] { 1, 0 },
            new double[] { 1, 1 }
        };
        double[] targets = { 0, 1, 1, 0 };

        for (int epoch = 0; epoch < 50000; epoch++)
        {
            double totalError = 0;

            for (int i = 0; i < inputs.Length; i++)
            {
                nn.Train(inputs[i], targets[i]);
                double output = nn.FeedForward(inputs[i]);
                double error = Math.Pow(targets[i] - output, 2);
                totalError += error;
            }

            Console.WriteLine($"Epoka {epoch + 1}, błąd sumaryczny: {Math.Round(totalError, 6)}");
        }

        Console.WriteLine("Wyniki sieci XOR po treningu:");
        for (int i = 0; i < inputs.Length; i++)
        {
            double output = nn.FeedForward(inputs[i]);
            Console.WriteLine($"{inputs[i][0]} XOR {inputs[i][1]} = {Math.Round(output, 4)}");
        }
    }
}
