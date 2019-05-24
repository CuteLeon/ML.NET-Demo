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

        public IEnumerable<House> GetDatas()
        => Enumerable.Range(10, 10).Select((index) =>
             {
                 float size = index;
                 float price = (float)(size * 150 * (1 + Random.Value.Next(-10, 10) / (double)100));
                 return new House(size, price);
             }).ToArray();
    }
}
