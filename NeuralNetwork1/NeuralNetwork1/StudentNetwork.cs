using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Diagnostics;
using System.Threading;

namespace NeuralNetwork1
{
    public class StudentNetwork : BaseNetwork
    {
        private Random rand;
        private double[][,] weights;
        private double[][] layers;
        private double[][] errors;
        public double learning_rate = 0.2;

        private double Sigmoid(double x)
        {
            return 1f / (1f + Math.Exp(-x));
        }

        private double SigmoidDerivative(double x)
        {
            return x * (1f - x);
        }

        /* перебирает все весовые матрицы и устанавливает их значения от -1 до 1.*/
        private void RandomizeWeights()
        {
           
                foreach (var w in weights) 
                {
                    for (int i = 0; i < w.GetLength(0); i++) 
                    {
                        for (int j = 0; j < w.GetLength(1); j++) 
                        {
                            w[i, j] = rand.NextDouble() * 2 - 1; 
                        }
                    }
               
            }
        }

        private void ForwardPass()
        {
              for (int k = 1; k < layers.Length; k++) 
                {
                    for (int j = 0; j < weights[k - 1].GetLength(1); j++) 
                    {
                        double sum = 0;
                        for (int i = 0; i < weights[k - 1].GetLength(0); i++) 
                        {
                            sum += weights[k - 1][i, j] * layers[k - 1][i]; 
                        }
                        layers[k][j] = Sigmoid(sum); 
                    }
                }
            
        }

        private void BackPropogation(int ans_idx)
        {
           
                int k = layers.Length - 1; 
                for (int j = 0; j < layers[k].Length; j++) 
                {
                    double n = layers[k][j];
                    errors[k][j] = -SigmoidDerivative(n) * ((j == ans_idx ? 1f : 0f) - n); 
                }

                
                for (k = layers.Length - 2; k > 0; k--)  
                {
                    for (int i = 0; i < layers[k].Length - 1; i++) 
                    {
                        for (int j = 0; j < weights[k].GetLength(1); j++) 
                        {
                            errors[k][i] += weights[k][i, j] * errors[k + 1][j];
                        }
                        errors[k][i] *= SigmoidDerivative(layers[k][i]); 
                    }
                }

           
                for (k = 0; k < weights.Length; k++)
                {
                    for (int i = 0; i < weights[k].GetLength(0); i++)
                    {
                        for (int j = 0; j < weights[k].GetLength(1); j++)
                        {
                            
                            weights[k][i, j] += -learning_rate * errors[k + 1][j] * layers[k][i];
                        }
                    }
                }
            
        }

        public StudentNetwork(int[] structure)
        {
            rand = new Random();

            layers = new double[structure.Length][];
            layers[0] = new double[structure[0] + 1]; 
            layers[0][structure[0]] = 1; // смещение для 1 слоя

            errors = new double[structure.Length][];

            weights = new double[structure.Length - 1][,]; 

        
            for (int i = 1; i < structure.Length; i++)
            {
                if (i == structure.Length - 1) 
                    layers[i] = new double[structure[i]]; // без bias тк выводной
                else 
                {
                    layers[i] = new double[structure[i] + 1]; 
                    layers[i][structure[i]] = 1; 
                }

                errors[i] = new double[structure[i]];

                if (i != structure.Length - 1)
                    layers[i][structure[i]] = 1;

                // Инициализация весов между текущим и предыдущим слоем
                weights[i - 1] = new double[structure[i - 1] + 1, structure[i]]; 
            }

            RandomizeWeights(); 

        }

        public override int Train(Sample sample, double acceptableError, bool parallel)
        {
         
            double est_error = 0; 
            int epochs = 0;

            while (true)
            {
       
                if (Predict(sample) == sample.actualClass && (est_error = sample.EstimatedError()) < acceptableError)
                {
                    return epochs; 
                }
                else
                {
                    epochs++; 
                    BackPropogation((int)sample.actualClass); 
                }
            }
        }
        
        public override double TrainOnDataSet(SamplesSet samplesSet, int epochsCount, double acceptableError, bool parallel)
        {
           
            double est_error = 0; // Оценка ошибки

            for (int epoch = 0; epoch < epochsCount; epoch++) 
            {
                est_error = 0;  
                foreach (var sample in samplesSet.samples) 
                {
                    Predict(sample);
                    est_error += sample.EstimatedError();
                    BackPropogation((int)sample.actualClass); // для корр весов
                }
                double error = est_error / samplesSet.samples.Count; // Средняя ошибка
                if (error < acceptableError) 
                {
                    return error; 
                }
            }
            return est_error / samplesSet.Count; 
        }

        // вычисляет вывод сети для заданного входа.
        protected override double[] Compute(double[] input)
        {
          
                for (int i = 0; i < input.Length; i++)
                {
                    layers[0][i] = input[i]; 
                }
            
            ForwardPass();
            return layers[layers.Length - 1];
        }

    }
}
