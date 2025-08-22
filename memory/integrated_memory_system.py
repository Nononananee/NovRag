# src/rag/integrated_memory_system.py
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
import asyncio
import json
import logging
from datetime import datetime, timedelta

from .memory_manager import NovelMemoryManager, MemoryChunk, MemoryType, MemoryPriority
from .emotional_memory_system import EmotionalMemorySystem, EmotionalState, EmotionalArc
from .chunking_strategies import NovelChunker, ChunkingStrategy
from .consistency_manager import LongTermConsistencyManager
from novel.structure.narrative_structure_manager import NarrativeStructureManager, NarrativeStructure
from novel.style.style_consistency_manager import StyleConsistencyManager, WritingStyle

logger = logging.getLogger(__name__)

@dataclass
class GenerationContext:
    """Complete context for novel generation including memory and consistency"""
    current_chapter: int
    current_word_count: int
    target_characters: List[str]
    active_plot_threads: List[str]
    generation_intent: str  # "continue", "dialogue", "scene", "character_intro"
    tone_requirements: Dict[str, float]
    constraints: Optional[Any] = None
    pov_character: Optional[str] = None
    scene_location: Optional[str] = None
    # Emotional context
    character_emotional_states: Optional[Dict[str, EmotionalState]] = None
    emotional_arc_requirements: Optional[List[str]] = None
    target_emotional_tone: Optional[str] = None
    # Narrative structure context
    narrative_structure: Optional[NarrativeStructure] = None
    expected_structure_stage: Optional[str] = None
    total_expected_words: Optional[int] = None
    # Style context
    target_writing_style: Optional[WritingStyle] = None
    style_consistency_required: bool = True

@dataclass
class GenerationResult:
    """Result of generation with memory and consistency metadata"""
    generated_content: str
    memory_chunks_created: List[str]  # Chunk IDs
    consistency_issues: List[Any]
    memory_tokens_used: int
    generation_metadata: Dict[str, Any]
    quality_score: float = 0.0
    originality_score: float = 0.0
    # Emotional analysis results
    emotional_states_detected: Optional[List[EmotionalState]] = None
    emotional_consistency_score: float = 0.0
    emotional_arc_progression: Optional[Dict[str, Any]] = None
    # Structure analysis results
    structure_validation_result: Optional[Dict[str, Any]] = None
    structure_adherence_score: float = 0.0
    # Style analysis results
    style_analysis_result: Optional[Dict[str, Any]] = None
    style_consistency_score: float = 0.0

class IntegratedNovelMemorySystem:
    """Integrated system combining memory management, chunking, and consistency"""
    
    def __init__(self, 
                 vectorstore_client=None,
                 llm_client=None,
                 character_repo=None,
                 db_utils=None,
                 max_memory_tokens: int = 32000,
                 consistency_level: str = "high"):
        
        # Core components
        self.memory_manager = NovelMemoryManager(
            max_context_tokens=max_memory_tokens,
            vectorstore_client=vectorstore_client
        )
        self.chunker = NovelChunker()
        self.consistency_manager = LongTermConsistencyManager(
            memory_manager=self.memory_manager,
            character_repo=character_repo
        )
        # Add emotional memory system
        self.emotional_memory = EmotionalMemorySystem(db_utils=db_utils)
        
        # Add narrative structure management
        self.structure_manager = NarrativeStructureManager(db_utils=db_utils)
        
        # Add style consistency management
        self.style_manager = StyleConsistencyManager(db_utils=db_utils)
        
        # Clients
        self.vectorstore = vectorstore_client
        self.llm_client = llm_client
        self.character_repo = character_repo  # Optional character repository
        
        # Configuration
        self.consistency_level = consistency_level
        self.max_memory_tokens = max_memory_tokens
        
        # Statistics
        self.stats = {
            'total_generations': 0,
            'consistency_issues_found': 0,
            'memory_chunks_created': 0,
            'average_context_size': 0,
            'emotional_states_analyzed': 0,
            'emotional_arcs_updated': 0,
            'average_emotional_consistency': 0.0,
            'structure_validations_performed': 0,
            'average_structure_adherence': 0.0,
            'style_analyses_performed': 0,
            'average_style_consistency': 0.0
        }

    async def generate_with_full_context(self, 
                                       generation_context: GenerationContext,
                                       prompt: str = None) -> GenerationResult:
        """Generate content with full memory and consistency context"""
        
        try:
            # Step 1: Enrich generation_context with the latest emotional states for validation later
            if generation_context.target_characters and self.emotional_memory.db_utils:
                try:
                    # Assuming the first character's document is the relevant one.
                    # This might need a more robust way to determine the current document context.
                    all_chunks = await self.memory_manager.get_all_memory_chunks()
                    if all_chunks:
                        # A simple way to get a relevant document_id
                        # In a real scenario, this should be passed in or determined more reliably
                        doc_id_for_context = all_chunks[0].document_id 
                        
                        latest_emotions_data = await self.emotional_memory.db_utils.get_latest_character_emotions(
                            document_id=doc_id_for_context,
                            character_names=generation_context.target_characters
                        )
                        latest_emotions = {
                            item['character_name']: EmotionalState(
                                character_name=item['character_name'],
                                primary_emotion=item['dominant_emotion'],
                                intensity=item['intensity']
                            ) for item in latest_emotions_data
                        }
                        generation_context.character_emotional_states = latest_emotions
                        logger.info(f"Enriched generation context with latest emotions for: {list(latest_emotions.keys())}")
                except Exception as e:
                    logger.error(f"Failed to enrich generation_context with emotions: {e}")

            # Step 2: Build optimal memory context
            memory_context = await self._build_memory_context(generation_context)
            
            # Step 3: Check consistency before generation
            consistency_ok, pre_issues = await self.consistency_manager.check_consistency_before_generation(
                context=memory_context,
                generation_intent=generation_context.__dict__
            )
            
            if not consistency_ok and self.consistency_level == "strict":
                return GenerationResult(
                    generated_content="",
                    memory_chunks_created=[],
                    consistency_issues=pre_issues,
                    memory_tokens_used=len(memory_context.split()),
                    generation_metadata={"error": "Critical consistency issues found"},
                    quality_score=0.0
                )
            
            # Step 4: Generate content with LLM
            generated_content = await self._generate_content_with_llm(
                context=memory_context,
                generation_context=generation_context,
                prompt=prompt
            )
            
            # Step 5: Validate generated content for consistency
            content_valid, post_issues = await self.consistency_manager.validate_generated_content(
                new_content=generated_content,
                context=memory_context
            )
            
            # Step 6: Analyze emotional content
            emotional_states = []
            emotional_consistency_score = 0.0
            emotional_arc_progression = {}
            
            try:
                if generation_context.target_characters:
                    emotional_states = await self.emotional_memory.analyze_emotional_content(
                        content=generated_content,
                        characters=generation_context.target_characters,
                        chunk_id=None,  # Will be set when chunk is created
                        method="keyword_analysis"
                    )
                    
                    # Calculate emotional consistency
                    emotional_consistency_score = await self._calculate_emotional_consistency(
                        emotional_states, generation_context
                    )
                    
                    # Track emotional arc progression
                    emotional_arc_progression = await self._track_emotional_arc_progression(
                        emotional_states, generation_context
                    )
                    
                    # Update statistics
                    self.stats['emotional_states_analyzed'] += len(emotional_states)
                    
            except Exception as e:
                logger.warning(f"Emotional analysis failed: {e}")
            
            # Step 7: Validate narrative structure
            structure_validation_result = None
            structure_adherence_score = 0.0
            
            try:
                if generation_context.narrative_structure and generation_context.total_expected_words:
                    # Set active structure if not already set
                    if self.structure_manager.active_structure is None:
                        self.structure_manager.set_active_structure(generation_context.narrative_structure)
                    
                    # Validate structure adherence
                    structure_validation = self.structure_manager.validate_structure_adherence(
                        current_chapter=generation_context.current_chapter,
                        current_word_count=generation_context.current_word_count,
                        total_expected_words=generation_context.total_expected_words,
                        plot_threads=generation_context.active_plot_threads,
                        character_arcs={}  # Could be extracted from character_repo
                    )
                    
                    structure_validation_result = {
                        'is_valid': structure_validation.is_valid,
                        'current_stage': structure_validation.current_stage.value,
                        'expected_stage': structure_validation.expected_stage.value,
                        'position_accuracy': structure_validation.position_accuracy,
                        'structural_issues': structure_validation.structural_issues,
                        'recommendations': structure_validation.recommendations
                    }
                    
                    structure_adherence_score = structure_validation.confidence_score
                    self.stats['structure_validations_performed'] += 1
                    
            except Exception as e:
                logger.warning(f"Structure validation failed: {e}")
            
            # Step 8: Analyze style consistency
            style_analysis_result = None
            style_consistency_score = 0.0
            
            try:
                if generation_context.style_consistency_required:
                    # Set active style guide if specified
                    if generation_context.target_writing_style:
                        self.style_manager.set_active_style_guide(generation_context.target_writing_style)
                    
                    # Analyze style consistency
                    style_analysis = self.style_manager.validate_style_consistency(generated_content)
                    
                    style_analysis_result = {
                        'consistency_score': style_analysis.consistency_score,
                        'style_deviations': style_analysis.style_deviations,
                        'recommendations': style_analysis.recommendations,
                        'confidence': style_analysis.confidence,
                        'analyzed_word_count': style_analysis.analyzed_word_count
                    }
                    
                    style_consistency_score = style_analysis.consistency_score
                    self.stats['style_analyses_performed'] += 1
                    
            except Exception as e:
                logger.warning(f"Style analysis failed: {e}")
            
            # Step 9: Fix consistency issues if possible
            if not content_valid:
                generated_content, remaining_issues = await self.consistency_manager.fix_consistency_issues(
                    content=generated_content,
                    issues=post_issues
                )
            else:
                remaining_issues = post_issues
            
            # Step 10: Process new content into memory chunks
            new_chunks = await self._process_new_content_into_memory(
                content=generated_content,
                generation_context=generation_context
            )
            
            # Step 11: Update memory system and emotional memory
            for chunk in new_chunks:
                await self.memory_manager.store_memory_chunk(chunk)
                
                # Update emotional states with chunk IDs
                for emotional_state in emotional_states:
                    if emotional_state.source_chunk_id is None:
                        emotional_state.source_chunk_id = chunk.id
            
            # Step 12: Calculate quality metrics
            quality_score = await self._calculate_content_quality(generated_content, memory_context)
            originality_score = await self._calculate_originality_score(generated_content)
            
            # Update statistics
            self._update_statistics(memory_context, new_chunks, remaining_issues)
            
            return GenerationResult(
                generated_content=generated_content,
                memory_chunks_created=[chunk.id for chunk in new_chunks],
                consistency_issues=remaining_issues,
                memory_tokens_used=len(memory_context.split()),
                generation_metadata={
                    "context_chunks_used": len(memory_context.split("===")), # Corrected escaping for

    async def _generate_content_with_llm(self, 
                                       context: str, 
                                       generation_context: GenerationContext,
                                       prompt: str = None) -> str:
        """Generate content using LLM with full context"""
        
        if not self.llm_client:
            return "LLM client not configured"
        
        # Build generation prompt
        system_prompt = self._build_system_prompt(generation_context)
        
        if prompt:
            user_prompt = f"{context}\n\n=== GENERATION REQUEST ===\n{prompt}"
        else:
            user_prompt = f"{context}\n\n=== GENERATION REQUEST ===\nContinue the novel from this point, maintaining consistency with all established elements."
        
        # Add constraints if specified
        if generation_context.constraints:
            constraint_text = self._format_constraints(generation_context.constraints)
            user_prompt = f"{user_prompt}\n\n=== CREATIVE CONSTRAINTS ===\n{constraint_text}"
        
        try:
            response = await self.llm_client.generate(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                max_tokens=2000,  # Configurable
                temperature=0.8,  # Configurable based on creativity needs
            )
            return response.strip()
        except Exception as e:
            return f"Generation failed: {str(e)}"

    async def _process_new_content_into_memory(self, 
                                             content: str, 
                                             generation_context: GenerationContext) -> List[MemoryChunk]:
        """Process newly generated content into memory chunks"""
        
        # Use adaptive chunking based on content characteristics
        novel_chunks = self.chunker.adaptive_chunking(
            text=content,
            context={
                'characters': generation_context.target_characters,
                'chapter': generation_context.current_chapter,
                'generation_type': generation_context.generation_intent
            }
        )
        
        # Convert to memory chunks with enhanced metadata
        memory_chunks = []
        for i, chunk in enumerate(novel_chunks):
            
            # Determine memory type based on chunk analysis
            memory_type = self._determine_memory_type(chunk)
            
            # Calculate priority based on content and context
            priority = self._calculate_chunk_priority(chunk, generation_context)
            
            # Create memory chunk
            memory_chunk = MemoryChunk(
                id=f"gen_{generation_context.current_chapter}_{i}_{datetime.now().timestamp()}",
                content=chunk.content,
                memory_type=memory_type,
                priority=priority,
                chapter_range=(generation_context.current_chapter, generation_context.current_chapter),
                word_range=(generation_context.current_word_count, 
                          generation_context.current_word_count + len(chunk.content.split())),
                characters_involved=chunk.narrative_elements.get('primary_characters', []),
                plot_threads=generation_context.active_plot_threads,
                tags=self._generate_chunk_tags(chunk, generation_context),
                embedding=None  # Will be generated when stored
            )
            
            memory_chunks.append(memory_chunk)
        
        return memory_chunks

    def _determine_memory_type(self, chunk) -> MemoryType:
        """Determine the memory type for a chunk"""
        
        chunk_type = chunk.chunk_type.lower()
        
        if 'dialogue' in chunk_type:
            return MemoryType.CHARACTER_STATE
        elif 'character' in chunk_type:
            return MemoryType.CHARACTER_STATE
        elif 'temporal' in chunk_type:
            return MemoryType.TIMELINE_EVENT
        elif chunk.importance_score > 0.8:
            return MemoryType.PLOT_POINT
        else:
            return MemoryType.WORLD_ELEMENT

    def _calculate_chunk_priority(self, chunk, generation_context: GenerationContext) -> MemoryPriority:
        """Calculate priority for memory chunk"""
        
        # Base priority on importance score
        importance = chunk.importance_score
        
        # Boost priority for key elements
        if any(char in generation_context.target_characters for char in 
               chunk.narrative_elements.get('primary_characters', [])):
            importance += 0.2
        
        if generation_context.generation_intent in ['climax', 'resolution', 'major_plot_point']:
            importance += 0.3
        
        # Convert to priority enum
        if importance >= 0.9:
            return MemoryPriority.CRITICAL
        elif importance >= 0.7:
            return MemoryPriority.HIGH
        elif importance >= 0.5:
            return MemoryPriority.MEDIUM
        else:
            return MemoryPriority.LOW

    def _generate_chunk_tags(self, chunk, generation_context: GenerationContext) -> List[str]:
        """Generate tags for memory chunk"""
        
        tags = []
        
        # Add chunk-based tags
        if hasattr(chunk, 'tags'):
            tags.extend(chunk.tags)
        
        # Add context-based tags
        tags.append(f"chapter_{generation_context.current_chapter}")
        tags.append(f"intent_{generation_context.generation_intent}")
        
        if generation_context.pov_character:
            tags.append(f"pov_{generation_context.pov_character}")
        
        if generation_context.scene_location:
            tags.append(f"location_{generation_context.scene_location}")
        
        # Add content-based tags
        content_lower = chunk.content.lower()
        if '"' in chunk.content or "'" in chunk.content:
            tags.append("dialogue")
        
        if any(word in content_lower for word in ['fight', 'battle', 'combat']):
            tags.append("action")
        
        if any(word in content_lower for word in ['love', 'kiss', 'romance']):
            tags.append("romance")
        
        if any(word in content_lower for word in ['magic', 'spell', 'enchant']):
            tags.append("magic")
        
        return list(set(tags))  # Remove duplicates

    async def _build_character_specific_context(self, character_id: str) -> str:
        """Build context specific to a POV character"""
        
        if not self.character_repo:
            return f"POV Character: {character_id}"
        
        try:
            character = await self.character_repo.get_character(character_id)
            if not character:
                return f"POV Character: {character_id} (not found in repository)"
            
            context_parts = [
                f"POV CHARACTER: {character.name}",
                f"Current emotional state: {character.get_current_emotional_state()}",
                f"Key personality traits: {', '.join(character.personality.get('traits', []))}",
                f"Current motivations: {', '.join(character.motivations)}",
                f"Speech patterns: {', '.join(character.speech_patterns)}"
            ]
            
            # Add recent character developments
            recent_developments = await self._get_recent_character_developments(character_id)
            if recent_developments:
                context_parts.append(f"Recent developments: {recent_developments}")
            
            return '\n'.join(context_parts)
            
        except Exception as e:
            return f"POV Character: {character_id} (error loading context: {e})"

    async def _build_location_context(self, location: str) -> str:
        """Build context for scene location"""
        
        # Get location-specific memories
        location_memories = await self._get_location_memories(location)
        
        context_parts = [f"LOCATION: {location}"]
        
        if location_memories:
            context_parts.append("Previous scenes at this location:")
            for memory in location_memories[:3]:  # Last 3 scenes at this location
                context_parts.append(f"- {memory['summary']}")
        
        return '\n'.join(context_parts)

    async def _calculate_content_quality(self, content: str, context: str) -> float:
        """Calculate quality score for generated content"""
        
        quality_factors = {
            'length_appropriate': len(content.split()) > 50,  # Not too short
            'character_consistency': True,  # Would implement actual check
            'narrative_flow': True,  # Would implement actual check
            'dialogue_quality': '"' in content,  # Has dialogue (if expected)
            'descriptive_balance': True  # Would implement actual check
        }
        
        # Calculate weighted score
        weights = {'length_appropriate': 0.2, 'character_consistency': 0.3, 
                  'narrative_flow': 0.3, 'dialogue_quality': 0.1, 'descriptive_balance': 0.1}
        
        score = sum(weights[factor] for factor, passed in quality_factors.items() if passed)
        return min(score, 1.0)

    async def _calculate_originality_score(self, content: str) -> float:
        """Calculate originality score for content"""
        
        # Simple originality metrics (would be more sophisticated)
        originality_factors = {
            'unique_phrases': len(set(content.lower().split())) / len(content.split()) if content else 0,
            'avoiding_cliches': not any(cliche in content.lower() 
                                      for cliche in ['suddenly', 'it was a dark', 'little did they know']),
            'creative_language': any(word in content.lower() 
                                   for word in ['shimmer', 'cascade', 'whisper', 'dance'])
        }
        
        # Weight factors
        weights = {'unique_phrases': 0.4, 'avoiding_cliches': 0.4, 'creative_language': 0.2}
        
        score = sum(weights[factor] * (1.0 if passed else 0.0) if isinstance(passed, bool) 
                   else weights[factor] * passed
                   for factor, passed in originality_factors.items())
        
        return min(score, 1.0)

    def _build_system_prompt(self, generation_context: GenerationContext) -> str:
        """Build system prompt for LLM generation"""
        
        prompt_parts = [
            "You are an expert novelist generating content for a long-form novel.",
            "Your task is to continue the story while maintaining perfect consistency with all established elements.",
            "",
            f"Current context: Chapter {generation_context.current_chapter}, ~{generation_context.current_word_count} words written",
            f"Generation intent: {generation_context.generation_intent}",
        ]
        
        if generation_context.target_characters:
            prompt_parts.append(f"Focus characters: {', '.join(generation_context.target_characters)}")
        
        if generation_context.pov_character:
            prompt_parts.append(f"Point of view: {generation_context.pov_character}")
        
        if generation_context.tone_requirements:
            tone_desc = ', '.join(f"{tone}: {score:.1f}" 
                                for tone, score in generation_context.tone_requirements.items())
            prompt_parts.append(f"Tone requirements: {tone_desc}")
        
        prompt_parts.extend([
            "",
            "Guidelines:",
            "- Maintain absolute consistency with established character traits, relationships, and world rules",
            "- Follow the narrative tone and style established in previous content",
            "- Avoid clichés and predictable plot developments",
            "- Show don't tell, use vivid sensory details",
            "- Ensure dialogue reflects each character's unique voice",
            "- Keep plot progression logical and motivated"
        ])
        
        return '\n'.join(prompt_parts)

    def _format_constraints(self, constraints) -> str:
        """Format creative constraints for LLM prompt"""
        
        if not constraints:
            return "No specific constraints"
        
        constraint_parts = []
        
        if hasattr(constraints, 'forbidden_tropes'):
            constraint_parts.append(f"Avoid these tropes: {', '.join(constraints.forbidden_tropes)}")
        
        if hasattr(constraints, 'style_requirements'):
            constraint_parts.append(f"Style requirements: {constraints.style_requirements}")
        
        if hasattr(constraints, 'originality_requirements'):
            constraint_parts.append(f"Originality level required: {constraints.originality_requirements}")
        
        return '\n'.join(constraint_parts) if constraint_parts else "Standard creative constraints apply"

    async def _get_recent_character_developments(self, character_id: str) -> str:
        """Get recent developments for a character"""
        
        # Get character-specific memory chunks from recent content
        if character_id in self.memory_manager.character_memories:
            recent_chunk_ids = self.memory_manager.character_memories[character_id][-3:]  # Last 3 chunks
            
            developments = []
            for chunk_id in recent_chunk_ids:
                if chunk_id in self.memory_manager.memory_chunks:
                    chunk = self.memory_manager.memory_chunks[chunk_id]
                    # Extract key developments (simplified)
                    developments.append(chunk.content[:100] + "...")
            
            return ' | '.join(developments)
        
        return ""

    async def _get_location_memories(self, location: str) -> List[Dict]:
        """Get memories associated with a location"""
        
        # Search for chunks with location tags
        location_memories = []
        
        for chunk in self.memory_manager.memory_chunks.values():
            if f"location_{location}" in chunk.tags:
                location_memories.append({
                    'summary': chunk.content[:150] + "...",
                    'chapter': chunk.chapter_range[0]
                })
        
        # Sort by recency
        location_memories.sort(key=lambda x: x['chapter'], reverse=True)
        
        return location_memories

    async def _calculate_emotional_consistency(self, emotional_states: List[EmotionalState], 
                                             generation_context: GenerationContext) -> float:
        """Calculate emotional consistency score for generated content"""
        
        if not emotional_states or not generation_context.character_emotional_states:
            return 0.0
        
        try:
            consistency_scores = []
            
            for emotional_state in emotional_states:
                character_name = emotional_state.character_name
                
                # Check if we have expected emotional state for this character
                if character_name in generation_context.character_emotional_states:
                    expected_state = generation_context.character_emotional_states[character_name]
                    
                    # Calculate similarity between expected and detected emotions
                    similarity = await self._calculate_emotional_similarity(
                        expected_state, emotional_state
                    )
                    consistency_scores.append(similarity)
            
            # Return average consistency score
            return sum(consistency_scores) / len(consistency_scores) if consistency_scores else 0.0
            
        except Exception as e:
            logger.warning(f"Error calculating emotional consistency: {e}")
            return 0.0

    async def _calculate_emotional_similarity(self, expected: EmotionalState, 
                                            detected: EmotionalState) -> float:
        """Calculate similarity between two emotional states"""
        
        try:
            # Simple similarity based on primary emotion and intensity
            if expected.primary_emotion == detected.primary_emotion:
                # Same emotion, check intensity difference
                intensity_diff = abs(expected.intensity - detected.intensity)
                return max(0.0, 1.0 - (intensity_diff / 10.0))  # Normalize to 0-1
            else:
                # Different emotions, check if they're related
                related_emotions = {
                    'joy': ['happiness', 'excitement', 'contentment'],
                    'sadness': ['grief', 'melancholy', 'despair'],
                    'anger': ['rage', 'frustration', 'irritation'],
                    'fear': ['anxiety', 'terror', 'worry'],
                    'surprise': ['shock', 'amazement', 'wonder'],
                    'disgust': ['revulsion', 'contempt', 'disdain']
                }
                
                for emotion_group in related_emotions.values():
                    if (expected.primary_emotion in emotion_group and 
                        detected.primary_emotion in emotion_group):
                        return 0.5  # Partial match for related emotions
                
                return 0.0  # No similarity
                
        except Exception as e:
            logger.warning(f"Error calculating emotional similarity: {e}")
            return 0.0

    async def _track_emotional_arc_progression(self, emotional_states: List[EmotionalState],
                                             generation_context: GenerationContext) -> Dict[str, Any]:
        """Track emotional arc progression for characters"""
        
        progression = {}
        
        try:
            for emotional_state in emotional_states:
                character_name = emotional_state.character_name
                
                if character_name not in progression:
                    progression[character_name] = {
                        'current_emotion': emotional_state.primary_emotion,
                        'intensity': emotional_state.intensity,
                        'arc_stage': self._determine_arc_stage(emotional_state, generation_context),
                        'progression_direction': self._determine_progression_direction(
                            emotional_state, generation_context
                        )
                    }
            
            # Update emotional arc statistics
            if progression:
                self.stats['emotional_arcs_updated'] += len(progression)
                
        except Exception as e:
            logger.warning(f"Error tracking emotional arc progression: {e}")
        
        return progression

    def _determine_arc_stage(self, emotional_state: EmotionalState, 
                           generation_context: GenerationContext) -> str:
        """Determine what stage of emotional arc the character is in"""
        
        # Simple heuristic based on chapter progress and emotion type
        chapter_progress = generation_context.current_chapter / 20.0  # Assume 20 chapter novel
        
        if chapter_progress < 0.25:
            return "setup"
        elif chapter_progress < 0.5:
            return "rising_action"
        elif chapter_progress < 0.75:
            return "climax"
        else:
            return "resolution"

    def _determine_progression_direction(self, emotional_state: EmotionalState,
                                       generation_context: GenerationContext) -> str:
        """Determine if character's emotional state is progressing positively or negatively"""
        
        positive_emotions = ['joy', 'happiness', 'excitement', 'contentment', 'love', 'hope']
        negative_emotions = ['sadness', 'anger', 'fear', 'disgust', 'despair', 'hatred']
        
        if emotional_state.primary_emotion in positive_emotions:
            return "positive"
        elif emotional_state.primary_emotion in negative_emotions:
            return "negative"
        else:
            return "neutral"

    def _update_statistics(self, memory_context: str, new_chunks: List[MemoryChunk], issues: List[Any]):
        """Update system statistics"""
        
        self.stats['total_generations'] += 1
        self.stats['consistency_issues_found'] += len(issues)
        self.stats['memory_chunks_created'] += len(new_chunks)
        
        # Update average context size
        context_size = len(memory_context.split())
        total_gens = self.stats['total_generations']
        current_avg = self.stats['average_context_size']
        self.stats['average_context_size'] = ((current_avg * (total_gens - 1)) + context_size) / total_gens

    def get_system_stats(self) -> Dict[str, Any]:
        """Get comprehensive system statistics"""
        
        memory_stats = self.memory_manager.get_memory_statistics()
        consistency_stats = self.consistency_manager.get_consistency_report()
        
        return {
            'memory_system': memory_stats,
            'consistency_system': consistency_stats,
            'generation_system': self.stats,
            'overall_health': {
                'memory_utilization': memory_stats['memory_usage']['active_context_chunks'] / 
                                    (memory_stats['total_chunks'] or 1),
                'consistency_compliance': 1.0 - (consistency_stats['critical_issues'] / 
                                                (consistency_stats['active_issues'] or 1)),
                'average_generation_quality': 0.85  # Would track actual quality scores
            }
        }

    async def cleanup_old_memories(self, days_old: int = 30, keep_critical: bool = True):
        """Clean up old, less relevant memories to manage storage"""
        
        cutoff_date = datetime.now() - timedelta(days=days_old)
        
        chunks_to_remove = []
        for chunk_id, chunk in self.memory_manager.memory_chunks.items():
            
            # Keep critical memories regardless of age
            if keep_critical and chunk.priority == MemoryPriority.CRITICAL:
                continue
            
            # Remove old, rarely accessed chunks
            if (chunk.last_accessed < cutoff_date and 
                chunk.access_count < 3 and 
                chunk.priority in [MemoryPriority.LOW, MemoryPriority.CONTEXTUAL]):
                
                chunks_to_remove.append(chunk_id)
        
        # Remove from memory and update indices
        for chunk_id in chunks_to_remove:
            if chunk_id in self.memory_manager.memory_chunks:
                del self.memory_manager.memory_chunks[chunk_id]
        
        # Update character and plot thread indices
        await self._rebuild_memory_indices()
        
        return len(chunks_to_remove)

    async def _rebuild_memory_indices(self):
        """Rebuild memory indices after cleanup"""
        
        # Clear existing indices
        self.memory_manager.character_memories.clear()
        self.memory_manager.plot_thread_memories.clear()
        self.memory_manager.chapter_index.clear()
        
        # Rebuild from remaining chunks
        for chunk_id, chunk in self.memory_manager.memory_chunks.items():
            
            # Update character index
            for character in chunk.characters_involved:
                if character not in self.memory_manager.character_memories:
                    self.memory_manager.character_memories[character] = []
                self.memory_manager.character_memories[character].append(chunk_id)
            
            # Update plot thread index
            for thread in chunk.plot_threads:
                if thread not in self.memory_manager.plot_thread_memories:
                    self.memory_manager.plot_thread_memories[thread] = []
                self.memory_manager.plot_thread_memories[thread].append(chunk_id)
            
            # Update chapter index
            for chapter in range(chunk.chapter_range[0], chunk.chapter_range[1] + 1):
                if chapter not in self.memory_manager.chapter_index:
                    self.memory_manager.chapter_index[chapter] = []
                self.memory_manager.chapter_index[chapter].append(chunk_id)