using System;
using System.Threading;

namespace ML.Utils
{
    /// <summary>
    /// 通用助手
    /// </summary>
    public static class Helper
    {
        /// <summary>
        /// 输出行
        /// </summary>
        /// <param name="message"></param>
        public static void PrintLine(string message)
            => Console.WriteLine($"{DateTime.Now.ToString("HH:mm:ss.fff")} threadId={Thread.CurrentThread.ManagedThreadId} : {message}");

        /// <summary>
        /// 输出分割线
        /// </summary>
        public static void PrintSplit()
            => Console.WriteLine("——————————————");

        /// <summary>
        /// 退出程序
        /// </summary>
        /// <param name="exitCode"></param>
        public static void Exit(int exitCode = 0)
        {
            PrintSplit();
            PrintLine($"程序即将退出... ExitCode={exitCode}");
            Console.Read();
            Environment.Exit(exitCode);
        }
    }
}
