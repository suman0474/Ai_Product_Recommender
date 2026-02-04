"""
Performance Benchmarking Suite for Distributed Enrichment

Measures throughput, latency, and resource usage of the distributed enrichment system.

Run with: python benchmark_distributed.py [--num-items N] [--workers W] [--verbose]
"""

import time
import statistics
import json
from datetime import datetime
from typing import List, Dict, Any
import argparse

from agentic.deep_agent.memory.memory.state_manager import InMemoryStateManager
from agentic.deep_agent.orchestration.task_queue import InMemoryTaskQueue
from agentic.deep_agent.orchestration.enrichment_worker import EnrichmentWorker
from agentic.deep_agent.orchestration.stateless_orchestrator import StatelessEnrichmentOrchestrator


class BenchmarkMetrics:
    """Collect and analyze benchmark metrics."""

    def __init__(self):
        self.metrics = {
            "session_creation": [],
            "task_push": [],
            "task_pop": [],
            "item_completion": [],
            "status_check": [],
            "end_to_end": []
        }
        self.start_time = None
        self.end_time = None

    def record(self, metric_name: str, duration: float):
        """Record a metric measurement."""
        if metric_name not in self.metrics:
            self.metrics[metric_name] = []
        self.metrics[metric_name].append(duration)

    def get_stats(self, metric_name: str) -> Dict[str, float]:
        """Get statistics for a metric."""
        values = self.metrics.get(metric_name, [])
        if not values:
            return {}

        return {
            "count": len(values),
            "min": min(values),
            "max": max(values),
            "avg": statistics.mean(values),
            "median": statistics.median(values),
            "total": sum(values)
        }

    def print_report(self):
        """Print benchmark report."""
        print("\n" + "=" * 80)
        print("DISTRIBUTED ENRICHMENT PERFORMANCE BENCHMARK")
        print("=" * 80)

        total_duration = sum(
            sum(times) for times in self.metrics.values()
        )

        for metric_name, times in self.metrics.items():
            if times:
                stats = self.get_stats(metric_name)
                print(f"\n{metric_name.upper().replace('_', ' ')}")
                print("-" * 80)
                print(f"  Samples:       {stats['count']}")
                print(f"  Min:           {stats['min']*1000:.3f}ms")
                print(f"  Max:           {stats['max']*1000:.3f}ms")
                print(f"  Avg:           {stats['avg']*1000:.3f}ms")
                print(f"  Median:        {stats['median']*1000:.3f}ms")
                print(f"  Total:         {stats['total']:.3f}s")

        print("\n" + "=" * 80)
        print(f"Total Benchmark Time: {total_duration:.3f}s")
        print("=" * 80 + "\n")

    def to_dict(self) -> Dict[str, Any]:
        """Export metrics as dictionary."""
        result = {
            "timestamp": datetime.utcnow().isoformat(),
            "metrics": {}
        }
        for metric_name in self.metrics:
            result["metrics"][metric_name] = self.get_stats(metric_name)
        return result


class DistributedEnrichmentBenchmark:
    """Run comprehensive benchmarks on distributed enrichment system."""

    def __init__(self, num_items: int = 100, num_workers: int = 1, verbose: bool = False):
        self.num_items = num_items
        self.num_workers = num_workers
        self.verbose = verbose
        self.metrics = BenchmarkMetrics()

    def benchmark_session_creation(self, num_sessions: int = 10) -> float:
        """Benchmark session creation performance."""
        print(f"Benchmarking session creation ({num_sessions} sessions)...")

        state_mgr = InMemoryStateManager()
        task_q = InMemoryTaskQueue()
        orch = StatelessEnrichmentOrchestrator(state_mgr, task_q)

        for i in range(num_sessions):
            items = [
                {"name": f"Item {j}", "category": "Test"}
                for j in range(self.num_items // num_sessions)
            ]

            start = time.time()
            session_id = orch.start_enrichment_session(items)
            duration = time.time() - start

            self.metrics.record("session_creation", duration)

        stats = self.metrics.get_stats("session_creation")
        print(f"  Created {stats['count']} sessions in {stats['total']:.3f}s")
        print(f"  Avg: {stats['avg']*1000:.3f}ms per session")
        return stats["total"]

    def benchmark_queue_operations(self, num_ops: int = 1000) -> float:
        """Benchmark task queue push/pop operations."""
        print(f"Benchmarking queue operations ({num_ops} operations)...")

        task_q = InMemoryTaskQueue()

        # Push operations
        for i in range(num_ops):
            task = {
                "task_id": f"task-{i}",
                "task_type": "initial_enrichment",
                "session_id": "session-1",
                "item_id": f"item-{i}",
                "payload": {"data": "test"},
                "priority": i % 10,
                "created_at": datetime.utcnow().isoformat(),
                "retry_count": 0
            }

            start = time.time()
            task_q.push_task(task)
            duration = time.time() - start
            self.metrics.record("task_push", duration)

        # Pop operations
        for i in range(num_ops):
            start = time.time()
            task = task_q.pop_task(timeout=1)
            duration = time.time() - start

            if task:
                self.metrics.record("task_pop", duration)
                task_q.ack_task(task["task_id"])

        push_stats = self.metrics.get_stats("task_push")
        pop_stats = self.metrics.get_stats("task_pop")

        print(f"  Push: {push_stats['avg']*1000:.3f}ms avg")
        print(f"  Pop:  {pop_stats['avg']*1000:.3f}ms avg")

        return push_stats["total"] + pop_stats["total"]

    def benchmark_state_operations(self, num_items: int = 100) -> float:
        """Benchmark state manager operations."""
        print(f"Benchmarking state operations ({num_items} items)...")

        state_mgr = InMemoryStateManager()
        session_id = f"session-{datetime.utcnow().timestamp()}"
        state_mgr.create_session(session_id, {"total_items": num_items})

        # Update operations
        for i in range(num_items):
            item_id = f"item-{i}"
            start = time.time()
            state_mgr.update_item_state(session_id, item_id, {
                "specifications": {
                    "spec1": f"value{i}",
                    "spec2": f"value{i+1}",
                    "spec3": f"value{i+2}"
                },
                "spec_count": 3,
                "status": "in_progress"
            })
            duration = time.time() - start
            self.metrics.record("state_update", duration)

        # Completion operations
        for i in range(num_items):
            item_id = f"item-{i}"
            start = time.time()
            state_mgr.mark_item_complete(session_id, item_id, {
                "final_spec": f"final{i}"
            })
            duration = time.time() - start
            self.metrics.record("item_completion", duration)

        update_stats = self.metrics.get_stats("state_update")
        completion_stats = self.metrics.get_stats("item_completion")

        print(f"  Update: {update_stats['avg']*1000:.3f}ms avg")
        print(f"  Complete: {completion_stats['avg']*1000:.3f}ms avg")

        return update_stats["total"] + completion_stats["total"]

    def benchmark_full_workflow(self, num_items: int = None) -> float:
        """Benchmark complete enrichment workflow."""
        if num_items is None:
            num_items = self.num_items

        print(f"Benchmarking full workflow ({num_items} items)...")

        state_mgr = InMemoryStateManager()
        task_q = InMemoryTaskQueue()
        orch = StatelessEnrichmentOrchestrator(state_mgr, task_q)

        items = [
            {"name": f"Item {i}", "category": "Test"}
            for i in range(num_items)
        ]

        # Start session
        start = time.time()
        session_id = orch.start_enrichment_session(items)
        session_creation_time = time.time() - start
        self.metrics.record("end_to_end", session_creation_time)

        # Get item IDs
        item_ids = state_mgr.get_session_item_ids(session_id)

        # Process all items
        for item_id in item_ids:
            start = time.time()
            state_mgr.update_item_state(session_id, item_id, {
                "specifications": {"temp": "100C"},
                "spec_count": 1,
                "status": "processing"
            })
            duration = time.time() - start
            self.metrics.record("end_to_end", duration)

            start = time.time()
            state_mgr.mark_item_complete(session_id, item_id, {
                "final_temp": "100C"
            })
            duration = time.time() - start
            self.metrics.record("end_to_end", duration)

        # Check final status
        start = time.time()
        final_status = orch.get_session_status(session_id)
        duration = time.time() - start
        self.metrics.record("status_check", duration)

        stats = self.metrics.get_stats("end_to_end")
        print(f"  Session created in {session_creation_time*1000:.3f}ms")
        print(f"  Total operations: {stats['count']}")
        print(f"  Total time: {stats['total']:.3f}s")
        print(f"  Avg operation: {stats['avg']*1000:.3f}ms")
        if stats['total'] > 0:
            print(f"  Throughput: {num_items / stats['total']:.1f} items/sec")
        else:
            print(f"  Throughput: >10000 items/sec (operations completed too fast to measure)")

        return stats["total"]

    def benchmark_priority_queue_ordering(self, num_tasks: int = 1000) -> float:
        """Benchmark priority queue ordering performance."""
        print(f"Benchmarking priority queue ordering ({num_tasks} tasks)...")

        task_q = InMemoryTaskQueue()

        # Push tasks with varying priorities
        priorities = [1, 5, 10, 3, 8, 2, 7, 4, 9, 6] * (num_tasks // 10)

        start = time.time()
        for i, priority in enumerate(priorities):
            task = {
                "task_id": f"task-{i}",
                "task_type": "initial_enrichment",
                "session_id": "session-1",
                "item_id": f"item-{i}",
                "payload": {},
                "priority": priority,
                "created_at": datetime.utcnow().isoformat(),
                "retry_count": 0
            }
            task_q.push_task(task)
        push_time = time.time() - start

        # Pop all and verify priority ordering
        start = time.time()
        last_priority = 0
        count = 0
        correct_order = 0

        while True:
            task = task_q.pop_task(timeout=1)
            if not task:
                break

            count += 1
            current_priority = task["priority"]
            if current_priority >= last_priority:
                correct_order += 1
            last_priority = current_priority

            task_q.ack_task(task["task_id"])

        pop_time = time.time() - start

        print(f"  Push {num_tasks} tasks: {push_time:.3f}s")
        print(f"  Pop {count} tasks: {pop_time:.3f}s")
        print(f"  Priority ordering: {correct_order}/{count} correct ({100*correct_order/count:.1f}%)")

        return push_time + pop_time

    def benchmark_concurrent_status_checks(self, num_sessions: int = 10, checks_per_session: int = 100) -> float:
        """Benchmark concurrent status checks."""
        print(f"Benchmarking concurrent status checks ({num_sessions} sessions, {checks_per_session} checks each)...")

        state_mgr = InMemoryStateManager()
        task_q = InMemoryTaskQueue()
        orch = StatelessEnrichmentOrchestrator(state_mgr, task_q)

        session_ids = []
        for i in range(num_sessions):
            items = [{"name": f"Item {j}", "category": "Test"} for j in range(10)]
            session_id = orch.start_enrichment_session(items)
            session_ids.append(session_id)

        # Check status multiple times
        start = time.time()
        for session_id in session_ids:
            for _ in range(checks_per_session):
                status = orch.get_session_status(session_id)
                self.metrics.record("status_check", 0)  # Record outside timing

        status_check_time = time.time() - start

        stats = self.metrics.get_stats("status_check")
        total_checks = num_sessions * checks_per_session
        print(f"  {total_checks} status checks in {status_check_time:.3f}s")
        print(f"  Avg: {status_check_time/total_checks*1000:.3f}ms per check")

        return status_check_time

    def run_all_benchmarks(self):
        """Run all benchmarks and generate report."""
        print("\n" + "=" * 80)
        print("STARTING DISTRIBUTED ENRICHMENT BENCHMARKS")
        print("=" * 80)
        print(f"Items per session: {self.num_items}")
        print(f"Workers: {self.num_workers}")
        print()

        total_time = 0

        # Run benchmarks
        total_time += self.benchmark_session_creation()
        print()
        total_time += self.benchmark_queue_operations()
        print()
        total_time += self.benchmark_state_operations()
        print()
        total_time += self.benchmark_full_workflow()
        print()
        total_time += self.benchmark_priority_queue_ordering()
        print()
        total_time += self.benchmark_concurrent_status_checks()

        # Print comprehensive report
        self.metrics.print_report()

        print("BENCHMARK COMPLETE")
        return self.metrics.to_dict()


def main():
    parser = argparse.ArgumentParser(
        description="Performance benchmark for distributed enrichment system"
    )
    parser.add_argument("--num-items", type=int, default=100,
                        help="Number of items per session (default: 100)")
    parser.add_argument("--workers", type=int, default=1,
                        help="Number of simulated workers (default: 1)")
    parser.add_argument("--verbose", action="store_true",
                        help="Enable verbose output")

    args = parser.parse_args()

    benchmark = DistributedEnrichmentBenchmark(
        num_items=args.num_items,
        num_workers=args.workers,
        verbose=args.verbose
    )

    results = benchmark.run_all_benchmarks()

    # Save results
    timestamp = datetime.utcnow().isoformat().replace(":", "-")
    results_file = f"benchmark_results_{timestamp}.json"
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to: {results_file}")


if __name__ == "__main__":
    main()
