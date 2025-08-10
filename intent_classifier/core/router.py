"""
Router module for orchestrating intent execution.
Routes intents to appropriate handlers and manages execution flow.
"""

import logging
import asyncio
from typing import List, Dict, Any, Optional, Tuple, Callable, Union
from dataclasses import dataclass, field
from enum import Enum
from concurrent.futures import ThreadPoolExecutor, as_completed
import time

from intent_classifier.models.schemas import (
    Intent, Composition, CompositionMode, JoinKey,
    IntentClassificationResult
)
from intent_classifier.core.taxonomy_loader import TaxonomyLoader


logger = logging.getLogger(__name__)


class ExecutionMode(Enum):
    """Execution mode for intent handling."""
    SYNC = "sync"
    ASYNC = "async"
    THREAD = "thread"


@dataclass
class ExecutionContext:
    """Context for intent execution."""
    intent: Intent
    clause_text: str
    position: int
    shared_entities: Dict[str, Any] = field(default_factory=dict)
    parent_results: Dict[int, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ExecutionResult:
    """Result from intent execution."""
    success: bool
    data: Any
    error: Optional[str] = None
    execution_time: float = 0.0
    warnings: List[str] = field(default_factory=list)


@dataclass
class RouteHandler:
    """Handler configuration for an intent route."""
    intent_pattern: str  # Can include wildcards, e.g., "DOC:*"
    handler: Callable
    execution_mode: ExecutionMode = ExecutionMode.SYNC
    timeout: float = 30.0
    retry_count: int = 0
    fallback_handler: Optional[Callable] = None


class IntentRouter:
    """
    Routes intents to appropriate handlers and manages execution.
    Supports sync/async execution, parallel processing, and error handling.
    """
    
    def __init__(self, taxonomy_loader: TaxonomyLoader,
                 max_workers: int = 4,
                 default_timeout: float = 30.0):
        """
        Initialize the router.
        
        Args:
            taxonomy_loader: Taxonomy loader instance
            max_workers: Maximum thread pool workers
            default_timeout: Default execution timeout
        """
        self.taxonomy = taxonomy_loader
        self.max_workers = max_workers
        self.default_timeout = default_timeout
        
        # Handler registry
        self.handlers: Dict[str, RouteHandler] = {}
        
        # Execution tracking
        self.execution_stats = {
            'total_executions': 0,
            'successful': 0,
            'failed': 0,
            'timeouts': 0,
            'average_time': 0.0
        }
        
        # Thread pool for parallel execution
        self.thread_pool = ThreadPoolExecutor(max_workers=max_workers)
        
        # Register default handlers
        self._register_default_handlers()
    
    def _register_default_handlers(self):
        """Register default intent handlers."""
        # Document retrieval handlers
        self.register_handler(
            "DOC:*",
            self._default_document_handler,
            ExecutionMode.THREAD
        )
        
        # Status check handlers
        self.register_handler(
            "STAT:*",
            self._default_status_handler,
            ExecutionMode.SYNC
        )
        
        # Count handlers
        self.register_handler(
            "COUNT:*",
            self._default_count_handler,
            ExecutionMode.SYNC
        )
        
        # Spec handlers
        self.register_handler(
            "SPEC:*",
            self._default_spec_handler,
            ExecutionMode.THREAD
        )
    
    def register_handler(self, intent_pattern: str, handler: Callable,
                        execution_mode: ExecutionMode = ExecutionMode.SYNC,
                        timeout: Optional[float] = None,
                        retry_count: int = 0,
                        fallback_handler: Optional[Callable] = None):
        """
        Register a handler for an intent pattern.
        
        Args:
            intent_pattern: Intent pattern (supports wildcards)
            handler: Handler function
            execution_mode: How to execute the handler
            timeout: Execution timeout
            retry_count: Number of retries on failure
            fallback_handler: Fallback if main handler fails
        """
        route_handler = RouteHandler(
            intent_pattern=intent_pattern,
            handler=handler,
            execution_mode=execution_mode,
            timeout=timeout or self.default_timeout,
            retry_count=retry_count,
            fallback_handler=fallback_handler
        )
        
        self.handlers[intent_pattern] = route_handler
        logger.info(f"Registered handler for pattern: {intent_pattern}")
    
    def route_single(self, intent: Intent, clause_text: str,
                    shared_entities: Optional[Dict[str, Any]] = None) -> ExecutionResult:
        """
        Route and execute a single intent.
        
        Args:
            intent: Intent to execute
            clause_text: Original clause text
            shared_entities: Entities from previous intents
            
        Returns:
            Execution result
        """
        context = ExecutionContext(
            intent=intent,
            clause_text=clause_text,
            position=0,
            shared_entities=shared_entities or {}
        )
        
        # Find matching handler
        handler = self._find_handler(intent.intent_code)
        if not handler:
            return ExecutionResult(
                success=False,
                data=None,
                error=f"No handler found for intent: {intent.intent_code}"
            )
        
        # Execute based on mode
        start_time = time.time()
        
        if handler.execution_mode == ExecutionMode.SYNC:
            result = self._execute_sync(handler, context)
        elif handler.execution_mode == ExecutionMode.ASYNC:
            result = self._execute_async(handler, context)
        elif handler.execution_mode == ExecutionMode.THREAD:
            result = self._execute_thread(handler, context)
        else:
            result = ExecutionResult(
                success=False,
                data=None,
                error=f"Unknown execution mode: {handler.execution_mode}"
            )
        
        result.execution_time = time.time() - start_time
        self._update_stats(result)
        
        return result
    
    def route_composition(self, intents: List[Intent], 
                         clauses: List[str],
                         composition: Composition) -> List[ExecutionResult]:
        """
        Route and execute a composition of intents.
        
        Args:
            intents: List of intents to execute
            clauses: List of clause texts
            composition: Composition plan
            
        Returns:
            List of execution results
        """
        results = []
        shared_entities = {}
        
        if composition.mode == CompositionMode.SINGLE:
            # Single intent
            result = self.route_single(intents[0], clauses[0])
            results.append(result)
            
        elif composition.mode == CompositionMode.SEQUENCE:
            # Sequential execution
            for idx in composition.ordering:
                # Gather entities from previous steps
                context_entities = shared_entities.copy()
                
                # Add entities based on join keys
                for jk in composition.join_keys:
                    if jk.to_step == idx and jk.from_step < len(results):
                        if results[jk.from_step].success:
                            from_data = results[jk.from_step].data
                            if isinstance(from_data, dict) and jk.key in from_data:
                                context_entities[jk.key] = from_data[jk.key]
                
                # Execute intent
                result = self.route_single(
                    intents[idx],
                    clauses[idx] if idx < len(clauses) else "",
                    context_entities
                )
                
                results.append(result)
                
                # Update shared entities
                if result.success and isinstance(result.data, dict):
                    shared_entities.update(result.data.get('entities', {}))
                
                # Stop on failure if required
                if not result.success and self._is_critical_intent(intents[idx]):
                    logger.error(f"Critical intent failed: {intents[idx].intent_code}")
                    break
                    
        elif composition.mode == CompositionMode.PARALLEL:
            # Parallel execution
            results = self._execute_parallel(intents, clauses, composition)
        
        return results
    
    def _find_handler(self, intent_code: str) -> Optional[RouteHandler]:
        """
        Find handler for an intent code.
        
        Args:
            intent_code: Intent code to match
            
        Returns:
            Matching handler or None
        """
        # Try exact match first
        if intent_code in self.handlers:
            return self.handlers[intent_code]
        
        # Try wildcard matches
        intent_prefix = intent_code.split(':')[0]
        
        # Check prefix wildcard
        prefix_pattern = f"{intent_prefix}:*"
        if prefix_pattern in self.handlers:
            return self.handlers[prefix_pattern]
        
        # Check full wildcard
        if "*" in self.handlers:
            return self.handlers["*"]
        
        return None
    
    def _execute_sync(self, handler: RouteHandler, 
                     context: ExecutionContext) -> ExecutionResult:
        """Execute handler synchronously."""
        try:
            data = handler.handler(context)
            return ExecutionResult(success=True, data=data)
        except Exception as e:
            logger.error(f"Sync execution failed: {e}")
            
            # Try fallback
            if handler.fallback_handler:
                try:
                    data = handler.fallback_handler(context)
                    return ExecutionResult(
                        success=True,
                        data=data,
                        warnings=[f"Used fallback handler: {str(e)}"]
                    )
                except Exception as fb_e:
                    logger.error(f"Fallback also failed: {fb_e}")
            
            return ExecutionResult(
                success=False,
                data=None,
                error=str(e)
            )
    
    def _execute_async(self, handler: RouteHandler,
                      context: ExecutionContext) -> ExecutionResult:
        """Execute handler asynchronously."""
        try:
            # Run async handler in event loop
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            
            async def run_async():
                return await handler.handler(context)
            
            data = loop.run_until_complete(
                asyncio.wait_for(run_async(), timeout=handler.timeout)
            )
            
            return ExecutionResult(success=True, data=data)
            
        except asyncio.TimeoutError:
            logger.error(f"Async execution timed out after {handler.timeout}s")
            self.execution_stats['timeouts'] += 1
            return ExecutionResult(
                success=False,
                data=None,
                error=f"Execution timed out after {handler.timeout}s"
            )
        except Exception as e:
            logger.error(f"Async execution failed: {e}")
            return ExecutionResult(
                success=False,
                data=None,
                error=str(e)
            )
        finally:
            loop.close()
    
    def _execute_thread(self, handler: RouteHandler,
                       context: ExecutionContext) -> ExecutionResult:
        """Execute handler in thread pool."""
        try:
            future = self.thread_pool.submit(handler.handler, context)
            data = future.result(timeout=handler.timeout)
            return ExecutionResult(success=True, data=data)
            
        except Exception as e:
            logger.error(f"Thread execution failed: {e}")
            return ExecutionResult(
                success=False,
                data=None,
                error=str(e)
            )
    
    def _execute_parallel(self, intents: List[Intent],
                         clauses: List[str],
                         composition: Composition) -> List[ExecutionResult]:
        """Execute intents in parallel."""
        futures = {}
        results = [None] * len(intents)
        
        # Submit all tasks
        for idx in composition.ordering:
            context = ExecutionContext(
                intent=intents[idx],
                clause_text=clauses[idx] if idx < len(clauses) else "",
                position=idx
            )
            
            handler = self._find_handler(intents[idx].intent_code)
            if handler:
                future = self.thread_pool.submit(
                    self._execute_handler_with_retry,
                    handler,
                    context
                )
                futures[future] = idx
            else:
                results[idx] = ExecutionResult(
                    success=False,
                    data=None,
                    error=f"No handler found for: {intents[idx].intent_code}"
                )
        
        # Collect results
        for future in as_completed(futures):
            idx = futures[future]
            try:
                results[idx] = future.result()
            except Exception as e:
                results[idx] = ExecutionResult(
                    success=False,
                    data=None,
                    error=str(e)
                )
        
        return results
    
    def _execute_handler_with_retry(self, handler: RouteHandler,
                                   context: ExecutionContext) -> ExecutionResult:
        """Execute handler with retry logic."""
        last_error = None
        
        for attempt in range(handler.retry_count + 1):
            if attempt > 0:
                logger.info(f"Retry attempt {attempt} for {context.intent.intent_code}")
                time.sleep(0.5 * attempt)  # Exponential backoff
            
            try:
                if handler.execution_mode == ExecutionMode.SYNC:
                    return self._execute_sync(handler, context)
                elif handler.execution_mode == ExecutionMode.ASYNC:
                    return self._execute_async(handler, context)
                elif handler.execution_mode == ExecutionMode.THREAD:
                    # Already in thread, execute directly
                    data = handler.handler(context)
                    return ExecutionResult(success=True, data=data)
                    
            except Exception as e:
                last_error = e
                logger.warning(f"Attempt {attempt + 1} failed: {e}")
        
        # All retries failed
        return ExecutionResult(
            success=False,
            data=None,
            error=f"Failed after {handler.retry_count + 1} attempts: {last_error}"
        )
    
    def _is_critical_intent(self, intent: Intent) -> bool:
        """Check if intent is critical (failure should stop execution)."""
        # Get routing hints
        critical = intent.routing_hints.get('critical', False)
        
        # Some intents are always critical
        critical_patterns = [
            'AUTH:*',  # Authentication
            'VALID:*',  # Validation
            'PERM:*'   # Permissions
        ]
        
        for pattern in critical_patterns:
            if self._matches_pattern(intent.intent_code, pattern):
                return True
        
        return critical
    
    def _matches_pattern(self, intent_code: str, pattern: str) -> bool:
        """Check if intent code matches pattern."""
        if pattern == intent_code:
            return True
        
        if pattern.endswith('*'):
            prefix = pattern[:-1]
            return intent_code.startswith(prefix)
        
        return False
    
    def _update_stats(self, result: ExecutionResult):
        """Update execution statistics."""
        self.execution_stats['total_executions'] += 1
        
        if result.success:
            self.execution_stats['successful'] += 1
        else:
            self.execution_stats['failed'] += 1
        
        # Update average time
        total = self.execution_stats['total_executions']
        avg = self.execution_stats['average_time']
        self.execution_stats['average_time'] = (
            (avg * (total - 1) + result.execution_time) / total
        )
    
    def get_stats(self) -> Dict[str, Any]:
        """Get execution statistics."""
        return self.execution_stats.copy()
    
    def shutdown(self):
        """Shutdown the router and clean up resources."""
        self.thread_pool.shutdown(wait=True)
        logger.info("Router shutdown complete")
    
    # Default handlers
    def _default_document_handler(self, context: ExecutionContext) -> Dict[str, Any]:
        """Default handler for document retrieval intents."""
        return {
            'intent': context.intent.intent_code,
            'entities': context.intent.entities,
            'message': f"Retrieved document for {context.intent.intent_code}",
            'data': {
                'document_id': context.intent.entities.get('ids.rfi') or 
                              context.intent.entities.get('ids.submittal') or
                              context.intent.entities.get('ids.cb'),
                'content': f"Mock content for {context.clause_text}"
            }
        }
    
    def _default_status_handler(self, context: ExecutionContext) -> Dict[str, Any]:
        """Default handler for status check intents."""
        return {
            'intent': context.intent.intent_code,
            'entities': context.intent.entities,
            'message': f"Status check for {context.intent.intent_code}",
            'data': {
                'status': 'pending',
                'last_updated': '2024-01-15',
                'details': f"Status details for {context.clause_text}"
            }
        }
    
    def _default_count_handler(self, context: ExecutionContext) -> Dict[str, Any]:
        """Default handler for count intents."""
        return {
            'intent': context.intent.intent_code,
            'entities': context.intent.entities,
            'message': f"Count for {context.intent.intent_code}",
            'data': {
                'count': 42,
                'breakdown': {
                    'open': 15,
                    'closed': 20,
                    'pending': 7
                }
            }
        }
    
    def _default_spec_handler(self, context: ExecutionContext) -> Dict[str, Any]:
        """Default handler for specification intents."""
        return {
            'intent': context.intent.intent_code,
            'entities': context.intent.entities,
            'message': f"Specification for {context.intent.intent_code}",
            'data': {
                'spec_section': context.intent.entities.get('spec_section', 'unknown'),
                'content': f"Specification content for {context.clause_text}",
                'version': '2.0'
            }
        }


class RouterBuilder:
    """Builder for creating configured routers."""
    
    @staticmethod
    def create_default_router(taxonomy_loader: TaxonomyLoader) -> IntentRouter:
        """
        Create a router with default configuration.
        
        Args:
            taxonomy_loader: Taxonomy loader instance
            
        Returns:
            Configured router
        """
        router = IntentRouter(taxonomy_loader)
        
        # Register additional default handlers
        # These would typically be replaced with actual implementations
        
        # Admin handlers
        router.register_handler(
            "ADMIN:*",
            lambda ctx: {
                'intent': ctx.intent.intent_code,
                'message': 'Admin action executed',
                'success': True
            },
            ExecutionMode.SYNC
        )
        
        # Response handlers
        router.register_handler(
            "RESP:*",
            lambda ctx: {
                'intent': ctx.intent.intent_code,
                'message': 'Response generated',
                'response': f"Response for {ctx.clause_text}"
            },
            ExecutionMode.SYNC
        )
        
        # Link handlers
        router.register_handler(
            "LINK:*",
            lambda ctx: {
                'intent': ctx.intent.intent_code,
                'message': 'Link established',
                'linked_items': []
            },
            ExecutionMode.THREAD
        )
        
        return router
    
    @staticmethod
    def create_async_router(taxonomy_loader: TaxonomyLoader) -> IntentRouter:
        """
        Create a router optimized for async operations.
        
        Args:
            taxonomy_loader: Taxonomy loader instance
            
        Returns:
            Async-optimized router
        """
        router = IntentRouter(
            taxonomy_loader,
            max_workers=8,  # More workers for async
            default_timeout=60.0  # Longer timeout
        )
        
        # Configure for async operations
        # (Would add async-specific handlers here)
        
        return router
