#!/usr/bin/env python3
"""
TIGER-LLM 推理服务测试客户端
"""
import argparse
import json
import time
import asyncio
import aiohttp
from typing import List, Dict
import random


class TIGERLLMClient:
    """TIGER-LLM API 客户端"""
    
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
    
    async def recommend(self, 
                       history: List[int], 
                       topk: int = 20,
                       user_id: str = None) -> Dict:
        """单用户推荐"""
        async with aiohttp.ClientSession() as session:
            payload = {
                "history": history,
                "topk": topk
            }
            if user_id:
                payload["user_id"] = user_id
            
            async with session.post(
                f"{self.base_url}/recommend",
                json=payload
            ) as resp:
                return await resp.json()
    
    async def recommend_batch(self, requests: List[Dict]) -> Dict:
        """批量推荐"""
        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{self.base_url}/recommend/batch",
                json={"requests": requests}
            ) as resp:
                return await resp.json()
    
    async def health(self) -> Dict:
        """健康检查"""
        async with aiohttp.ClientSession() as session:
            async with session.get(f"{self.base_url}/health") as resp:
                return await resp.json()
    
    async def benchmark(self, num_requests: int = 100) -> Dict:
        """性能测试"""
        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{self.base_url}/benchmark?num_requests={num_requests}"
            ) as resp:
                return await resp.json()


async def test_single_recommendation(client: TIGERLLMClient):
    """测试单用户推荐"""
    print("\n=== 测试单用户推荐 ===")
    
    history = [1, 2, 3, 4, 5]
    
    start = time.time()
    result = await client.recommend(history, topk=10, user_id="test_user")
    latency = (time.time() - start) * 1000
    
    print(f"延迟: {latency:.2f} ms")
    print(f"后端: {result.get('backend', 'unknown')}")
    print(f"推理时间: {result.get('inference_time_ms', 0):.2f} ms")
    print("推荐结果:")
    for rec in result.get('recommendations', [])[:5]:
        print(f"  Item {rec['item_id']}: {rec['score']:.4f}")


async def test_batch_recommendation(client: TIGERLLMClient):
    """测试批量推荐"""
    print("\n=== 测试批量推荐 ===")
    
    # 构造 10 个用户的请求
    requests = []
    for i in range(10):
        history = [random.randint(1, 1000) for _ in range(random.randint(5, 15))]
        requests.append({
            "user_id": f"user_{i}",
            "history": history,
            "topk": 10
        })
    
    start = time.time()
    result = await client.recommend_batch(requests)
    latency = (time.time() - start) * 1000
    
    print(f"批量处理 10 个用户")
    print(f"总延迟: {latency:.2f} ms")
    print(f"平均每个用户: {latency/10:.2f} ms")


async def test_concurrent_requests(client: TIGERLLMClient, num_requests: int = 100):
    """测试并发性能"""
    print(f"\n=== 测试并发性能 ({num_requests} 请求) ===")
    
    async def single_request():
        history = [random.randint(1, 1000) for _ in range(10)]
        return await client.recommend(history, topk=10)
    
    start = time.time()
    tasks = [single_request() for _ in range(num_requests)]
    results = await asyncio.gather(*tasks)
    total_time = time.time() - start
    
    latencies = [r.get('inference_time_ms', 0) for r in results]
    avg_latency = sum(latencies) / len(latencies)
    
    print(f"总时间: {total_time:.2f} s")
    print(f"QPS: {num_requests/total_time:.2f}")
    print(f"平均延迟: {avg_latency:.2f} ms")
    print(f"P99 延迟: {sorted(latencies)[int(len(latencies)*0.99)]:.2f} ms")


async def main():
    parser = argparse.ArgumentParser(description='TIGER-LLM Client')
    parser.add_argument('--url', type=str, default='http://localhost:8000',
                       help='Server URL')
    parser.add_argument('--test', type=str, default='all',
                       choices=['single', 'batch', 'concurrent', 'all'],
                       help='Test type')
    parser.add_argument('--num-requests', type=int, default=100,
                       help='Number of requests for concurrent test')
    
    args = parser.parse_args()
    
    client = TIGERLLMClient(args.url)
    
    # 健康检查
    print("=== 健康检查 ===")
    try:
        health = await client.health()
        print(f"状态: {health['status']}")
        print(f"后端: {health['backend']}")
        print(f"最大批处理: {health['max_batch_size']}")
    except Exception as e:
        print(f"❌ 无法连接服务器: {e}")
        return
    
    # 运行测试
    if args.test in ['single', 'all']:
        await test_single_recommendation(client)
    
    if args.test in ['batch', 'all']:
        await test_batch_recommendation(client)
    
    if args.test in ['concurrent', 'all']:
        await test_concurrent_requests(client, args.num_requests)
    
    print("\n✓ 测试完成!")


if __name__ == '__main__':
    asyncio.run(main())
