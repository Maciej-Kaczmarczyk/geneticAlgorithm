using System;
using System.Collections.Generic;
using System.Linq;

namespace GeneticAlgorithm
{
    class Program
    {
        const int ChromosomesPerParameter = 3; 
        const double MutationRate = 0.1; 
        const int TournamentSize = 2; 

        const double MinValue = 0;
        const double MaxValue = 100;

        static Random random = new Random();

        class IndividualFitness
        {
            public int[] Individual { get; set; }
            public double Fitness { get; set; }
        }

        static double EvaluateFitness(int[] individual)
        {
            var (x1, x2) = DecodeIndividual(individual);
            return Math.Sin(x1 * 0.05) + Math.Sin(x2 * 0.05) + 0.4 * Math.Sin(x1 * 0.15) * Math.Sin(x2 * 0.15);
        }

        static (double x1, double x2) DecodeIndividual(int[] individual)
        {
            double x1 = DecodeChromosome(individual.Take(ChromosomesPerParameter).ToArray());
            double x2 = DecodeChromosome(individual.Skip(ChromosomesPerParameter).ToArray());
            return (x1, x2);
        }

        static double DecodeChromosome(int[] chromosome)
        {
            int value = 0;
            for (int i = 0; i < chromosome.Length; i++)
            {
                value += chromosome[i] * (int)Math.Pow(2, i);
            }
            return MinValue + (MaxValue - MinValue) * value / (Math.Pow(2, chromosome.Length) - 1);
        }

        static int[] TournamentSelection(List<IndividualFitness> evaluatedPopulation)
        {
            var tournament = evaluatedPopulation.OrderBy(x => random.Next()).Take(TournamentSize).ToList();
            return tournament.OrderByDescending(ind => ind.Fitness).First().Individual;
        }

        static int[] Mutate(int[] individual)
        {
            var mutatedIndividual = individual.ToArray();
            if (random.NextDouble() < MutationRate)
            {
                int mutationPoint = random.Next(mutatedIndividual.Length);
                mutatedIndividual[mutationPoint] = 1 - mutatedIndividual[mutationPoint];
            }
            return mutatedIndividual;
        }
    }
}