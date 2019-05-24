namespace ML.NET_Demo.Models
{
    /// <summary>
    /// 房屋
    /// </summary>
    public class House
    {
        /// <summary>
        /// 构造函数
        /// </summary>
        /// <param name="size"></param>
        /// <param name="price"></param>
        public House(float size, float price)
        {
            this.Size = size;
            this.Price = price;
        }

        /// <summary>
        /// 面积
        /// </summary>
        public float Size { get; set; }

        /// <summary>
        /// 价格
        /// </summary>
        public float Price { get; set; }
    }
}
