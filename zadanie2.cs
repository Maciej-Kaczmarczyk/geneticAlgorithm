using System;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Linq;

namespace GeneticAlgorithmApproximation
{
    class Program
    {
        const int PopulationSize = 13;
        const int ChromosomesPerParameter = 4;
        const int Iterations = 100;
        const double MutationRate = 0.1;
        const int TournamentSize = 3;
        const double MinValue = 0.0;
        const double MaxValue = 3.0;

        static Random random = new Random();

        static List<(double x, double y)> Samples;

        class IndividualFitness
        {
            public int[] Chromosomes;
            public double Fitness;
        }

        static void Main(string[] args)
        {
            LoadSamples("sinusik.txt");

            var population = InitializePopulation();

            for (int iteration = 0; iteration <= Iterations; iteration++)
            {
                var evaluated = population.Select(p => new IndividualFitness
                {
                    Chromosomes = p,
                    Fitness = EvaluateFitness(p)
                }).ToList();

                var bestFitness = evaluated.Max(e => e.Fitness);
                var avgFitness = evaluated.Average(e => e.Fitness);

                Console.WriteLine($"Iteracja {iteration}: Najlepsze przystosowanie = {bestFitness:F6}, Średnie = {avgFitness:F6}");

                if (iteration == Iterations) break;

                var newPopulation = new List<int[]>();

                // 4 osobniki: selekcja + krzyżowanie
                for (int i = 0; i < 4; i += 2)
                {
                    var parent1 = TournamentSelection(evaluated);
                    var parent2 = TournamentSelection(evaluated);
                    var (child1, child2) = Crossover(parent1, parent2);
                    newPopulation.Add(child1);
                    newPopulation.Add(child2);
                }

                // 4 osobniki: selekcja + mutacja
                for (int i = 0; i < 4; i++)
                {
                    var parent = TournamentSelection(evaluated);
                    var mutant = Mutate(parent);
                    newPopulation.Add(mutant);
                }

                // 4 osobniki: selekcja + krzyżowanie + mutacja
                for (int i = 0; i < 2; i++)
                {
                    var parent1 = TournamentSelection(evaluated);
                    var parent2 = TournamentSelection(evaluated);
                    var (child1, child2) = Crossover(parent1, parent2);
                    newPopulation.Add(Mutate(child1));
                    newPopulation.Add(Mutate(child2));
                }

                // elitaryzm
                var best = evaluated.OrderByDescending(e => e.Fitness).First().Chromosomes;
                newPopulation.Add(best);

                population = newPopulation.ToArray();
            }

            var bestFinal = population.OrderByDescending(p => EvaluateFitness(p)).First();
            var (pa, pb, pc) = DecodeIndividual(bestFinal);
            Console.WriteLine($"\nNajlepszy wynik końcowy: pa = {pa:F4}, pb = {pb:F4}, pc = {pc:F4}, fitness = {EvaluateFitness(bestFinal):F6}");
        }

        static void LoadSamples(string path)
        {
            Samples = File.ReadAllLines(path)
                .Where(line => !string.IsNullOrWhiteSpace(line))
                .Select(line =>
                {
                    var parts = line.Trim().Split(new[] { ' ', '\t' }, StringSplitOptions.RemoveEmptyEntries);

                    return (double.Parse(parts[0], CultureInfo.InvariantCulture), double.Parse(parts[1], CultureInfo.InvariantCulture));

                }).ToList();
        }

        static int[][] InitializePopulation()
        {
            return Enumerable.Range(0, PopulationSize)
                .Select(_ => Enumerable.Range(0, ChromosomesPerParameter * 3)
                    .Select(__ => random.Next(2)).ToArray())
                .ToArray();
        }

        static double EvaluateFitness(int[] chromosomes)
        {
            var (pa, pb, pc) = DecodeIndividual(chromosomes);
            return -Samples.Sum(s =>
            {
                var predicted = pa * Math.Sin(pb * s.x + pc);
                return Math.Pow(predicted - s.y, 2);
            });
        }

        static (double, double, double) DecodeIndividual(int[] individual)
        {
            double Decode(int[] bits)
            {
                int value = bits.Select((bit, index) => bit * (1 << index)).Sum();
                return MinValue + (MaxValue - MinValue) * value / (Math.Pow(2, bits.Length) - 1);
            }

            var pa = Decode(individual.Take(ChromosomesPerParameter).ToArray());
            var pb = Decode(individual.Skip(ChromosomesPerParameter).Take(ChromosomesPerParameter).ToArray());
            var pc = Decode(individual.Skip(2 * ChromosomesPerParameter).Take(ChromosomesPerParameter).ToArray());
            return (pa, pb, pc);
        }

        static int[] TournamentSelection(List<IndividualFitness> population)
        {
            return population.OrderBy(_ => random.Next())
                             .Take(TournamentSize)
                             .OrderByDescending(ind => ind.Fitness)
                             .First().Chromosomes;
        }

        static int[] Mutate(int[] individual)
        {
            var copy = individual.ToArray();
            if (random.NextDouble() < MutationRate)
            {
                int index = random.Next(copy.Length);
                copy[index] = 1 - copy[index];
            }
            return copy;
        }

        static (int[], int[]) Crossover(int[] parent1, int[] parent2)
        {
            int point = random.Next(1, parent1.Length - 1);
            var child1 = parent1.Take(point).Concat(parent2.Skip(point)).ToArray();
            var child2 = parent2.Take(point).Concat(parent1.Skip(point)).ToArray();
            return (child1, child2);
        }
    }
}
