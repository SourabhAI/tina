"""
Tests for the router module.
"""

import pytest
import asyncio
import time
from unittest.mock import Mock, patch
from concurrent.futures import TimeoutError as FutureTimeoutError

from intent_classifier.core.router import (
    IntentRouter, ExecutionMode, ExecutionContext, ExecutionResult,
    RouteHandler, RouterBuilder
)
from intent_classifier.models.schemas import (
    Intent, Composition, CompositionMode, JoinKey
)
from intent_classifier.core.taxonomy_loader import TaxonomyLoader


class TestIntentRouter:
    """Test the IntentRouter class."""
    
    @pytest.fixture
    def mock_taxonomy(self):
        """Create a mock taxonomy loader."""
        mock = Mock(spec=TaxonomyLoader)
        return mock
    
    @pytest.fixture
    def router(self, mock_taxonomy):
        """Create router instance."""
        return IntentRouter(mock_taxonomy, max_workers=2)
    
    def test_initialization(self, router):
        """Test router initialization."""
        assert router.max_workers == 2
        assert router.default_timeout == 30.0
        assert len(router.handlers) > 0  # Default handlers registered
        assert router.execution_stats['total_executions'] == 0
    
    def test_register_handler(self, router):
        """Test handler registration."""
        def test_handler(ctx):
            return {"test": "data"}
        
        router.register_handler(
            "TEST:*",
            test_handler,
            ExecutionMode.SYNC,
            timeout=10.0,
            retry_count=2
        )
        
        assert "TEST:*" in router.handlers
        handler = router.handlers["TEST:*"]
        assert handler.handler == test_handler
        assert handler.execution_mode == ExecutionMode.SYNC
        assert handler.timeout == 10.0
        assert handler.retry_count == 2
    
    def test_find_handler(self, router):
        """Test handler matching logic."""
        # Register test handlers
        router.register_handler("DOC:RFI_RETRIEVE", lambda x: x)
        router.register_handler("STAT:*", lambda x: x)
        router.register_handler("*", lambda x: x)
        
        # Test exact match
        handler = router._find_handler("DOC:RFI_RETRIEVE")
        assert handler is not None
        assert handler.intent_pattern == "DOC:RFI_RETRIEVE"
        
        # Test wildcard match
        handler = router._find_handler("STAT:RFI_STATUS")
        assert handler is not None
        assert handler.intent_pattern == "STAT:*"
        
        # Test fallback to full wildcard
        handler = router._find_handler("UNKNOWN:SOMETHING")
        assert handler is not None
        assert handler.intent_pattern == "*"
    
    def test_route_single_sync(self, router):
        """Test single intent routing with sync handler."""
        def sync_handler(ctx):
            return {
                "intent": ctx.intent.intent_code,
                "data": "sync result"
            }
        
        router.register_handler("TEST:SYNC", sync_handler, ExecutionMode.SYNC)
        
        intent = Intent(
            coarse_class="TEST",
            intent_code="TEST:SYNC",
            confidence=0.9,
            entities={},
            policies={},
            routing_hints={}
        )
        
        result = router.route_single(intent, "test clause")
        
        assert result.success == True
        assert result.data["data"] == "sync result"
        assert result.execution_time > 0
        assert router.execution_stats['successful'] == 1
    
    def test_route_single_async(self, router):
        """Test single intent routing with async handler."""
        async def async_handler(ctx):
            await asyncio.sleep(0.1)
            return {
                "intent": ctx.intent.intent_code,
                "data": "async result"
            }
        
        router.register_handler("TEST:ASYNC", async_handler, ExecutionMode.ASYNC)
        
        intent = Intent(
            coarse_class="TEST",
            intent_code="TEST:ASYNC",
            confidence=0.9,
            entities={},
            policies={},
            routing_hints={}
        )
        
        result = router.route_single(intent, "test clause")
        
        assert result.success == True
        assert result.data["data"] == "async result"
        assert result.execution_time >= 0.1
    
    def test_route_single_thread(self, router):
        """Test single intent routing with thread handler."""
        def thread_handler(ctx):
            time.sleep(0.1)
            return {
                "intent": ctx.intent.intent_code,
                "data": "thread result"
            }
        
        router.register_handler("TEST:THREAD", thread_handler, ExecutionMode.THREAD)
        
        intent = Intent(
            coarse_class="TEST",
            intent_code="TEST:THREAD",
            confidence=0.9,
            entities={},
            policies={},
            routing_hints={}
        )
        
        result = router.route_single(intent, "test clause")
        
        assert result.success == True
        assert result.data["data"] == "thread result"
        assert result.execution_time >= 0.1
    
    def test_route_single_with_error(self, router):
        """Test error handling in routing."""
        def error_handler(ctx):
            raise ValueError("Test error")
        
        router.register_handler("TEST:ERROR", error_handler)
        
        intent = Intent(
            coarse_class="TEST",
            intent_code="TEST:ERROR",
            confidence=0.9,
            entities={},
            policies={},
            routing_hints={}
        )
        
        result = router.route_single(intent, "test clause")
        
        assert result.success == False
        assert "Test error" in result.error
        assert router.execution_stats['failed'] == 1
    
    def test_route_single_with_fallback(self, router):
        """Test fallback handler."""
        def main_handler(ctx):
            raise ValueError("Main failed")
        
        def fallback_handler(ctx):
            return {"data": "fallback result"}
        
        router.register_handler(
            "TEST:FALLBACK",
            main_handler,
            fallback_handler=fallback_handler
        )
        
        intent = Intent(
            coarse_class="TEST",
            intent_code="TEST:FALLBACK",
            confidence=0.9,
            entities={},
            policies={},
            routing_hints={}
        )
        
        result = router.route_single(intent, "test clause")
        
        assert result.success == True
        assert result.data["data"] == "fallback result"
        assert len(result.warnings) > 0
    
    def test_route_single_no_handler(self, router):
        """Test routing with no matching handler."""
        intent = Intent(
            coarse_class="UNKNOWN",
            intent_code="UNKNOWN:INTENT",
            confidence=0.9,
            entities={},
            policies={},
            routing_hints={}
        )
        
        result = router.route_single(intent, "test clause")
        
        assert result.success == False
        assert "No handler found" in result.error
    
    def test_route_composition_single(self, router):
        """Test composition routing for single intent."""
        intents = [
            Intent(
                coarse_class="DOC",
                intent_code="DOC:RFI_RETRIEVE",
                confidence=0.9,
                entities={"ids.rfi": 123},
                policies={},
                routing_hints={}
            )
        ]
        
        composition = Composition(
            mode=CompositionMode.SINGLE,
            ordering=[0],
            join_keys=[],
            response_policy="single_message"
        )
        
        results = router.route_composition(intents, ["Find RFI 123"], composition)
        
        assert len(results) == 1
        assert results[0].success == True
        assert results[0].data["intent"] == "DOC:RFI_RETRIEVE"
    
    def test_route_composition_sequence(self, router):
        """Test sequential composition routing."""
        intents = [
            Intent(
                coarse_class="DOC",
                intent_code="DOC:RFI_RETRIEVE",
                confidence=0.9,
                entities={"ids.rfi": 123},
                policies={},
                routing_hints={}
            ),
            Intent(
                coarse_class="STAT",
                intent_code="STAT:RFI_STATUS",
                confidence=0.85,
                entities={},
                policies={},
                routing_hints={}
            )
        ]
        
        composition = Composition(
            mode=CompositionMode.SEQUENCE,
            ordering=[0, 1],
            join_keys=[JoinKey(from_step=0, to_step=1, key="ids.rfi")],
            response_policy="step_by_step"
        )
        
        # Mock handlers to test entity passing
        def doc_handler(ctx):
            return {
                "entities": {"ids.rfi": 123},
                "data": "document data"
            }
        
        def stat_handler(ctx):
            # Should receive RFI ID from previous step
            return {
                "received_rfi": ctx.shared_entities.get("ids.rfi"),
                "status": "open"
            }
        
        router.register_handler("DOC:RFI_RETRIEVE", doc_handler)
        router.register_handler("STAT:RFI_STATUS", stat_handler)
        
        results = router.route_composition(
            intents,
            ["Find RFI 123", "check status"],
            composition
        )
        
        assert len(results) == 2
        assert results[0].success == True
        assert results[1].success == True
        assert results[1].data["received_rfi"] == 123
    
    def test_route_composition_parallel(self, router):
        """Test parallel composition routing."""
        intents = [
            Intent(
                coarse_class="DOC",
                intent_code="DOC:SUBMITTAL_RETRIEVE",
                confidence=0.9,
                entities={},
                policies={},
                routing_hints={}
            ),
            Intent(
                coarse_class="COUNT",
                intent_code="COUNT:RFI_COUNT",
                confidence=0.85,
                entities={},
                policies={},
                routing_hints={}
            )
        ]
        
        composition = Composition(
            mode=CompositionMode.PARALLEL,
            ordering=[0, 1],
            join_keys=[],
            response_policy="separate_messages"
        )
        
        results = router.route_composition(
            intents,
            ["Show submittals", "count RFIs"],
            composition
        )
        
        assert len(results) == 2
        assert all(r.success for r in results)
    
    def test_critical_intent_detection(self, router):
        """Test critical intent detection."""
        # Test with routing hint
        intent1 = Intent(
            coarse_class="DOC",
            intent_code="DOC:TEST",
            confidence=0.9,
            entities={},
            policies={},
            routing_hints={"critical": True}
        )
        assert router._is_critical_intent(intent1) == True
        
        # Test with pattern
        intent2 = Intent(
            coarse_class="AUTH",
            intent_code="AUTH:LOGIN",
            confidence=0.9,
            entities={},
            policies={},
            routing_hints={}
        )
        assert router._is_critical_intent(intent2) == True
        
        # Test non-critical
        intent3 = Intent(
            coarse_class="DOC",
            intent_code="DOC:TEST",
            confidence=0.9,
            entities={},
            policies={},
            routing_hints={}
        )
        assert router._is_critical_intent(intent3) == False
    
    def test_retry_logic(self, router):
        """Test retry mechanism."""
        attempt_count = 0
        
        def flaky_handler(ctx):
            nonlocal attempt_count
            attempt_count += 1
            if attempt_count < 3:
                raise ValueError(f"Attempt {attempt_count} failed")
            return {"success": True}
        
        router.register_handler(
            "TEST:RETRY",
            flaky_handler,
            retry_count=2
        )
        
        intent = Intent(
            coarse_class="TEST",
            intent_code="TEST:RETRY",
            confidence=0.9,
            entities={},
            policies={},
            routing_hints={}
        )
        
        result = router.route_single(intent, "test")
        
        assert result.success == True
        assert attempt_count == 3
    
    def test_timeout_handling(self, router):
        """Test timeout handling."""
        async def slow_handler(ctx):
            await asyncio.sleep(2.0)
            return {"data": "slow"}
        
        router.register_handler(
            "TEST:TIMEOUT",
            slow_handler,
            ExecutionMode.ASYNC,
            timeout=0.5
        )
        
        intent = Intent(
            coarse_class="TEST",
            intent_code="TEST:TIMEOUT",
            confidence=0.9,
            entities={},
            policies={},
            routing_hints={}
        )
        
        result = router.route_single(intent, "test")
        
        assert result.success == False
        assert "timed out" in result.error
        assert router.execution_stats['timeouts'] == 1
    
    def test_execution_stats(self, router):
        """Test execution statistics tracking."""
        # Run some successful and failed executions
        def good_handler(ctx):
            time.sleep(0.1)
            return {"ok": True}
        
        def bad_handler(ctx):
            raise ValueError("Bad")
        
        router.register_handler("TEST:GOOD", good_handler)
        router.register_handler("TEST:BAD", bad_handler)
        
        good_intent = Intent("TEST", "TEST:GOOD", 0.9, {}, {}, {})
        bad_intent = Intent("TEST", "TEST:BAD", 0.9, {}, {}, {})
        
        # Execute multiple times
        router.route_single(good_intent, "test")
        router.route_single(good_intent, "test")
        router.route_single(bad_intent, "test")
        
        stats = router.get_stats()
        
        assert stats['total_executions'] == 3
        assert stats['successful'] == 2
        assert stats['failed'] == 1
        assert stats['average_time'] > 0.05  # At least half the sleep time
    
    def test_shutdown(self, router):
        """Test router shutdown."""
        router.shutdown()
        # Should not raise any errors
        # Thread pool should be shut down


class TestRouterBuilder:
    """Test the RouterBuilder class."""
    
    @pytest.fixture
    def mock_taxonomy(self):
        """Create a mock taxonomy loader."""
        return Mock(spec=TaxonomyLoader)
    
    def test_create_default_router(self, mock_taxonomy):
        """Test default router creation."""
        router = RouterBuilder.create_default_router(mock_taxonomy)
        
        assert isinstance(router, IntentRouter)
        assert router.max_workers == 4
        assert router.default_timeout == 30.0
        
        # Check that additional handlers are registered
        assert "ADMIN:*" in router.handlers
        assert "RESP:*" in router.handlers
        assert "LINK:*" in router.handlers
    
    def test_create_async_router(self, mock_taxonomy):
        """Test async router creation."""
        router = RouterBuilder.create_async_router(mock_taxonomy)
        
        assert isinstance(router, IntentRouter)
        assert router.max_workers == 8
        assert router.default_timeout == 60.0
    
    def test_default_handlers(self, mock_taxonomy):
        """Test that default handlers work correctly."""
        router = RouterBuilder.create_default_router(mock_taxonomy)
        
        # Test document handler
        doc_intent = Intent(
            coarse_class="DOC",
            intent_code="DOC:RFI_RETRIEVE",
            confidence=0.9,
            entities={"ids.rfi": 123},
            policies={},
            routing_hints={}
        )
        
        result = router.route_single(doc_intent, "Find RFI 123")
        assert result.success == True
        assert result.data["data"]["document_id"] == 123
        
        # Test status handler
        stat_intent = Intent(
            coarse_class="STAT",
            intent_code="STAT:RFI_STATUS",
            confidence=0.9,
            entities={},
            policies={},
            routing_hints={}
        )
        
        result = router.route_single(stat_intent, "Check status")
        assert result.success == True
        assert result.data["data"]["status"] == "pending"
        
        # Test count handler
        count_intent = Intent(
            coarse_class="COUNT",
            intent_code="COUNT:RFI_COUNT",
            confidence=0.9,
            entities={},
            policies={},
            routing_hints={}
        )
        
        result = router.route_single(count_intent, "Count RFIs")
        assert result.success == True
        assert result.data["data"]["count"] == 42
        
        # Cleanup
        router.shutdown()
