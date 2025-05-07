using System;
using System.Collections.Generic;
using System.Linq;

namespace GeneticAlgorithmXOR
{
    class Program
    {
        const int PopulationSize = 13;
        const int ChromosomesPerWeight = 8;
        const int WeightCount = 9;
        const int Iterations = 1000;
        const int TournamentSize = 3;
        const double MutationRate = 0.1;
        const double MinWeight = -10.0;
        const double MaxWeight = 10.0;

        static Random random = new Random();

        class IndividualFitness
        {
            public int[] Chromosomes;
            public double Fitness;
        }

        static readonly (double[] input, double expected)[] xorDataset =
        {
            (new double[] { 0, 0 }, 0),
            (new double[] { 0, 1 }, 1),
            (new double[] { 1, 0 }, 1),
            (new double[] { 1, 1 }, 0)
        };

        static void Main()
        {
            var population = InitializePopulation();

            for (int iteration = 0; iteration <= Iterations; iteration++)
            {
                var evaluated = population.Select(p => new IndividualFitness
                {
                    Chromosomes = p,
                    Fitness = -EvaluateFitness(p)
                }).ToList();

                var bestFitness = -evaluated.Max(e => e.Fitness);
                var avgFitness = -evaluated.Average(e => e.Fitness);
                Console.WriteLine($"Iteracja {iteration}: Najlepszy błąd = {bestFitness:F6}, Średni błąd = {avgFitness:F6}");

                if (iteration == Iterations) break;

                var newPopulation = new List<int[]>();

                for (int i = 0; i < 4; i += 2)
                {
                    var p1 = TournamentSelection(evaluated);
                    var p2 = TournamentSelection(evaluated);
                    var (c1, c2) = Crossover(p1, p2);
                    newPopulation.Add(c1);
                    newPopulation.Add(c2);
                }

                for (int i = 0; i < 4; i++)
                {
                    var p = TournamentSelection(evaluated);
                    newPopulation.Add(Mutate(p));
                }

                for (int i = 0; i < 2; i++)
                {
                    var p1 = TournamentSelection(evaluated);
                    var p2 = TournamentSelection(evaluated);
                    var (c1, c2) = Crossover(p1, p2);
                    newPopulation.Add(Mutate(c1));
                    newPopulation.Add(Mutate(c2));
                }

                var best = evaluated.OrderByDescending(e => e.Fitness).First().Chromosomes;
                newPopulation.Add(best);

                population = newPopulation.ToArray();
            }

            var finalBest = population.OrderBy(p => EvaluateFitness(p)).First();
            var weights = DecodeWeights(finalBest);
            Console.WriteLine("\nNajlepsze wagi:");
            for (int i = 0; i < weights.Length; i++)
                Console.WriteLine($"w{i} = {weights[i]:F4}");

            Console.WriteLine($"Końcowy błąd: {EvaluateFitness(finalBest):F6}");
        }

        static int[][] InitializePopulation()
        {
            return Enumerable.Range(0, PopulationSize)
                .Select(_ => Enumerable.Range(0, ChromosomesPerWeight * WeightCount)
                .Select(__ => random.Next(2)).ToArray())
                .ToArray();
        }

        static double EvaluateFitness(int[] chromos)
        {
            double[] weights = DecodeWeights(chromos);

            double TotalError = 0;
            foreach (var (input, expected) in xorDataset)
            {
                double[] hidden = new double[2];

                for (int i = 0; i < 2; i++)
                {
                    hidden[i] = Sigmoid(
                        input[0] * weights[i * 2] +
                        input[1] * weights[i * 2 + 1] +
                        weights[4 + i]
                    );
                }

                double output = Sigmoid(hidden[0] * weights[6] + hidden[1] * weights[7] + weights[8]);
                TotalError += Math.Pow(output - expected, 2);
            }

            return TotalError;
        }

        static double[] DecodeWeights(int[] individual)
        {
            double[] weights = new double[WeightCount];
            for (int i = 0; i < WeightCount; i++)
            {
                int start = i * ChromosomesPerWeight;
                int value = 0;
                for (int j = 0; j < ChromosomesPerWeight; j++)
                    value += individual[start + j] << j;

                weights[i] = MinWeight + (MaxWeight - MinWeight) * value / (Math.Pow(2, ChromosomesPerWeight) - 1);
            }
            return weights;
        }

        static double Sigmoid(double x) => 1.0 / (1.0 + Math.Exp(-x));

        static int[] TournamentSelection(List<IndividualFitness> pop)
        {
            return pop.OrderBy(_ => random.Next())
                      .Take(TournamentSize)
                      .OrderByDescending(x => x.Fitness)
                      .First().Chromosomes;
        }

        static int[] Mutate(int[] chrom)
        {
            var copy = chrom.ToArray();
            if (random.NextDouble() < MutationRate)
            {
                int index = random.Next(copy.Length);
                copy[index] = 1 - copy[index];
            }
            return copy;
        }

        static (int[], int[]) Crossover(int[] p1, int[] p2)
        {
            int cut = random.Next(1, p1.Length - 1);
            return (
                p1.Take(cut).Concat(p2.Skip(cut)).ToArray(),
                p2.Take(cut).Concat(p1.Skip(cut)).ToArray()
            );
        }
    }
}
