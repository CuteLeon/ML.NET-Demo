using System.Linq;
using Microsoft.ML.Probabilistic;
using Microsoft.ML.Probabilistic.Distributions;
using Microsoft.ML.Probabilistic.Models;
using ML.Utils;

namespace InferNet
{
    class Program
    {
        static void Main(string[] args)
        {
            Helper.PrintLine("Infer.NET 概率编程");

            Helper.PrintLine("生成比赛数据...");
            var winnerData = new[] { 0, 0, 0, 1, 3, 4 };
            var loserData = new[] { 1, 3, 4, 2, 1, 2 };
            var game = new Range(winnerData.Length);
            var player = new Range(winnerData.Concat(loserData).Max() + 1);
            var playerSkills = Variable.Array<double>(player);
            playerSkills[player] = Variable.GaussianFromMeanAndVariance(6, 9).ForEach(player);
            var winners = Variable.Array<int>(game);
            var losers = Variable.Array<int>(game);

            Helper.PrintLine("模拟比赛...");
            using (Variable.ForEach(game))
            {
                var winnerPerformance = Variable.GaussianFromMeanAndVariance(playerSkills[winners[game]], 1.0);
                var loserPerformance = Variable.GaussianFromMeanAndVariance(playerSkills[losers[game]], 1.0);

                // 约束为真
                Variable.ConstrainTrue(winnerPerformance > loserPerformance);
            }

            Helper.PrintLine("附加数据...");
            winners.ObservedValue = winnerData;
            losers.ObservedValue = loserData;

            Helper.PrintLine("运行概率引擎...");
            var inferenceEngine = new InferenceEngine();
            var inferredSkills = inferenceEngine.Infer<Gaussian[]>(playerSkills);

            var orderedPlayerSkills = inferredSkills
                .Select((s, i) => new { Player = i, Skill = s })
                .OrderByDescending(ps => ps.Skill.GetMean());

            foreach (var playerSkill in orderedPlayerSkills)
            {
                Helper.PrintLine($"Player {playerSkill.Player} skill: {playerSkill.Skill}");
            }

            System.Console.Read();
        }
    }
}
