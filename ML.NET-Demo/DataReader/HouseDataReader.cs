using System;
using System.Collections.Generic;
using System.Linq;
using ML.NET_Demo.Models;

namespace ML.NET_Demo.DataReader
{
    /// <summary>
    /// 数据读取器
    /// </summary>
    public class HouseDataReader
    {
        protected static Lazy<Random> Random = new Lazy<Random>(() => new Random(), true);

        public IEnumerable<House> GetTrainingDatas()
        => Enumerable.Range(0, 1000).Select((index) =>
             {
                 float size = (float)(10 + index / 100.0d);
                 float price = (float)(this.GetPrice(size) * (1 + Random.Value.Next(-3, 3) / (double)100));
                 return new House(size, price);
             }).ToArray();

        public IEnumerable<House> GetTestDatas()
        => Enumerable.Range(10, 20).Select(Index => new House(Index, this.GetPrice(Index)));

        private float GetPrice(float size)
            => size * 150;
    }
}
