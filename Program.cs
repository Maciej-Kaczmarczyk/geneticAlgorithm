using System;
using System.Collections.Generic;
using System.Linq;

namespace GeneticAlgorithmExample
{
    class Program
    {
        // Parametry algorytmu
        const int PopulationSize = 9; // Liczba osobników w populacji (nieparzysta)
        const int ChromosomesPerParameter = 3; // Liczba chromosomów na parametr (co najmniej 3)
        const int Iterations = 20; // Liczba iteracji algorytmu
        const double MutationRate = 0.1; // Prawdopodobieństwo mutacji
        const int TournamentSize = 2; // Rozmiar turnieju (2-20% liczby osobników)

        // Zakres wartości parametrów
        const double MinValue = 0;
        const double MaxValue = 100;

        static Random random = new Random();

        // Klasa pomocnicza do przechowywania osobnika i jego przystosowania
        class IndividualFitness
        {
            public int[] Individual { get; set; }
            public double Fitness { get; set; }
        }

        static void Main(string[] args)
        {
            // Inicjalizacja populacji
            var population = InitializePopulation();

            for (int iteration = 0; iteration < Iterations; iteration++)
            {
                // Ocena populacji
                var evaluatedPopulation = population.Select(ind => new IndividualFitness
                {
                    Individual = ind,
                    Fitness = EvaluateFitness(ind)
                }).ToList();

                // Wypisanie najlepszego i średniego przystosowania
                var bestFitness = evaluatedPopulation.Max(ind => ind.Fitness);
                var averageFitness = evaluatedPopulation.Average(ind => ind.Fitness);
                Console.WriteLine($"Iteracja {iteration + 1}: Najlepsze przystosowanie = {bestFitness}, Średnie przystosowanie = {averageFitness}");

                // Selekcja, mutacja i tworzenie nowej populacji
                var newPopulation = new int[PopulationSize][];
                for (int i = 0; i < PopulationSize - 1; i++)
                {
                    var selectedIndividual = TournamentSelection(evaluatedPopulation);
                    newPopulation[i] = Mutate(selectedIndividual); // Mutacja wybranego osobnika
                }

                // Dodanie najlepszego osobnika z poprzedniej populacji (elitaryzm)
                var bestIndividual = evaluatedPopulation.OrderByDescending(ind => ind.Fitness).First().Individual;
                newPopulation[PopulationSize - 1] = bestIndividual;

                // Zastąpienie starej populacji nową
                population = newPopulation;
            }

            // Wynik końcowy
            var finalBestIndividual = population.OrderByDescending(ind => EvaluateFitness(ind)).First();
            var (x1, x2) = DecodeIndividual(finalBestIndividual);
            Console.WriteLine($"Najlepsze rozwiązanie: x1 = {x1}, x2 = {x2}, Wartość funkcji = {EvaluateFitness(finalBestIndividual)}");
        }

        // Inicjalizacja populacji
        static int[][] InitializePopulation()
        {
            var population = new int[PopulationSize][];
            for (int i = 0; i < PopulationSize; i++)
            {
                population[i] = new int[ChromosomesPerParameter * 2]; // 2 parametry: x1 i x2
                for (int j = 0; j < ChromosomesPerParameter * 2; j++)
                {
                    population[i][j] = random.Next(2); // Losowe bity (0 lub 1)
                }
            }
            return population;
        }

        // Ocena przystosowania osobnika
        static double EvaluateFitness(int[] individual)
        {
            var (x1, x2) = DecodeIndividual(individual);
            return Math.Sin(x1 * 0.05) + Math.Sin(x2 * 0.05) + 0.4 * Math.Sin(x1 * 0.15) * Math.Sin(x2 * 0.15);
        }

        // Dekodowanie osobnika na wartości x1 i x2
        static (double x1, double x2) DecodeIndividual(int[] individual)
        {
            double x1 = DecodeChromosome(individual.Take(ChromosomesPerParameter).ToArray());
            double x2 = DecodeChromosome(individual.Skip(ChromosomesPerParameter).ToArray());
            return (x1, x2);
        }

        // Dekodowanie chromosomu na wartość liczbową
        static double DecodeChromosome(int[] chromosome)
        {
            int value = 0;
            for (int i = 0; i < chromosome.Length; i++)
            {
                value += chromosome[i] * (int)Math.Pow(2, i);
            }
            return MinValue + (MaxValue - MinValue) * value / (Math.Pow(2, chromosome.Length) - 1);
        }

        // Selekcja turniejowa
        static int[] TournamentSelection(List<IndividualFitness> evaluatedPopulation)
        {
            var tournament = evaluatedPopulation.OrderBy(x => random.Next()).Take(TournamentSize).ToList();
            return tournament.OrderByDescending(ind => ind.Fitness).First().Individual;
        }

        // Mutacja jednopunktowa
        static int[] Mutate(int[] individual)
        {
            var mutatedIndividual = individual.ToArray(); // Tworzymy kopię osobnika
            if (random.NextDouble() < MutationRate)
            {
                int mutationPoint = random.Next(mutatedIndividual.Length);
                mutatedIndividual[mutationPoint] = 1 - mutatedIndividual[mutationPoint]; // Odwrócenie bitu
            }
            return mutatedIndividual;
        }
    }
}