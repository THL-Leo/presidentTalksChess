import chess
import chess.engine
import random
from typing import List, Dict, Tuple
from dataclasses import dataclass
import json
import openai
import os
import asyncio
import logging
from time import sleep
from dotenv import load_dotenv

# Enable debug logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Load environment variables from .env file
load_dotenv()

@dataclass
class CommentaryEvent:
    timestamp: float
    text: str
    priority: int  # 1=low, 2=medium, 3=high
    move_san: str = ""
    evaluation: float = 0.0

class ChessCommentator:
    def __init__(self, engine_path: str = None, openai_api_key: str = None):
        # Set engine path from environment or parameter
        self.engine_path = (
            engine_path or 
            os.getenv("STOCKFISH_PATH") or 
            "./stockfish/stockfish-macos-m1-apple-silicon"
        )
        
        self.engine = None
        self.previous_eval = 0.0
        self.move_count = 0
        
        # OpenAI configuration from environment variables
        self.openai_model = os.getenv("OPENAI_MODEL", "gpt-3.5-turbo")
        self.openai_temperature = float(os.getenv("OPENAI_TEMPERATURE", "0.8"))
        self.openai_max_tokens = int(os.getenv("OPENAI_MAX_TOKENS", "100"))
        
        # Stockfish configuration from environment variables
        self.stockfish_depth = int(os.getenv("STOCKFISH_DEPTH", "10"))
        self.stockfish_time_limit = float(os.getenv("STOCKFISH_TIME_LIMIT", "1.0"))
        
        # Setup OpenAI API
        self.openai_client = None
        self.setup_openai_client(openai_api_key)
        
        # Game context for better commentary
        self.game_phase = "opening"  # opening, middlegame, endgame
        self.last_commentary = ""
        self.move_history = []

    def setup_openai_client(self, api_key: str = None):
        """Setup OpenAI client with API key"""
        try:
            # Try to get API key from parameter, environment, or prompt user
            if api_key:
                openai.api_key = api_key
            elif os.getenv("OPENAI_API_KEY"):
                openai.api_key = os.getenv("OPENAI_API_KEY")
            else:
                print("⚠️  OpenAI API key not found!")
                print("💡 Set OPENAI_API_KEY environment variable or pass api_key parameter")
                print("   Example: export OPENAI_API_KEY='your-api-key-here'")
                return False
            
            # Test the API key with a simple request
            self.openai_client = openai
            print("✅ OpenAI API client initialized successfully")
            return True
            
        except Exception as e:
            print(f"❌ Failed to setup OpenAI client: {e}")
            return False

    def determine_game_phase(self, board: chess.Board) -> str:
        """Determine current game phase based on pieces on board"""
        piece_count = len(board.piece_map())
        
        if self.move_count < 10:
            return "opening"
        elif piece_count > 20:
            return "middlegame"
        elif piece_count <= 12:
            return "endgame"
        else:
            return "middlegame"

    def generate_gpt_commentary(self, board: chess.Board, move: chess.Move, 
                               analysis: Dict, move_quality: str, 
                               previous_board: chess.Board = None) -> str:
        """Generate commentary using GPT"""
        if not self.openai_client:
            return "Commentary unavailable - OpenAI API not configured."
        
        try:
            # Prepare context for GPT
            try:
                move_san = previous_board.san(move) if previous_board and move else "Game start"
            except (ValueError, chess.IllegalMoveError):
                # Fallback if move is not legal for the board
                move_san = str(move) if move else "Game start"
            
            current_eval = analysis.get("evaluation", 0.0)
            move_type = self.detect_move_type(board, move) if move else "start"
            
            # Update game phase
            self.game_phase = self.determine_game_phase(board)
            
            # Create detailed prompt for GPT
            prompt = self.create_commentary_prompt(
                move_san, move_type, move_quality, current_eval, 
                self.previous_eval, self.game_phase, board
            )
            
            # Make API call to GPT
            response = self.openai_client.chat.completions.create(
                model=self.openai_model,
                messages=[
                    {"role": "system", "content": "You are an entertaining chess commentator. Provide engaging, natural commentary about chess moves. Keep responses to 1-2 sentences, be conversational and enthusiastic."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=self.openai_max_tokens,
                temperature=self.openai_temperature,
                top_p=0.9
            )
            
            commentary = response.choices[0].message.content.strip()
            
            # Store for context in next commentary
            self.last_commentary = commentary
            self.move_history.append({
                "move": move_san,
                "evaluation": current_eval,
                "commentary": commentary
            })
            
            # Keep only last 5 moves for context
            if len(self.move_history) > 5:
                self.move_history = self.move_history[-5:]
            
            return commentary
            
        except Exception as e:
            print(f"Error generating GPT commentary: {e}")
            # Fallback to simple description
            return f"The move is {move_san}. Current evaluation: {current_eval:.1f}"

    def create_commentary_prompt(self, move_san: str, move_type: str, 
                                move_quality: str, current_eval: float, 
                                previous_eval: float, game_phase: str, 
                                board: chess.Board) -> str:
        """Create a detailed prompt for GPT commentary generation"""
        
        # Calculate evaluation change
        eval_change = current_eval - previous_eval
        
        # Determine position characteristics
        piece_count = len(board.piece_map())
        in_check = board.is_check()
        
        # Build context about recent moves
        recent_context = ""
        if len(self.move_history) > 0:
            recent_moves = [f"{h['move']} (eval: {h['evaluation']:.1f})" 
                          for h in self.move_history[-2:]]
            recent_context = f"Recent moves: {', '.join(recent_moves)}. "
        
        prompt = f"""
Chess Commentary Request:

Move: {move_san}
Move Type: {move_type} ({move_quality} quality)
Game Phase: {game_phase}
Position: {"In check" if in_check else "Normal position"}
Pieces remaining: {piece_count}

Evaluation:
- Current: {current_eval:.1f}
- Previous: {previous_eval:.1f}
- Change: {eval_change:+.1f} ({"+" if eval_change >= 0 else ""}{eval_change:.1f})

{recent_context}

Generate engaging chess commentary for this move. Consider:
- The evaluation change and what it means
- The type of move and game phase
- Be enthusiastic and natural
- Keep it to 1-2 sentences
- Don't repeat recent commentary themes

Commentary:"""

        return prompt

    async def start_engine(self):
        """Initialize the chess engine asynchronously"""
        try:
            logger.info(f"Starting Stockfish engine at: {self.engine_path}")
            self.transport, self.engine = await chess.engine.popen_uci(self.engine_path)
            logger.info("Stockfish engine started successfully")
            print("Stockfish engine started successfully")
            return True
        except Exception as e:
            logger.error(f"Failed to start engine: {e}")
            print(f"Failed to start engine: {e}")
            self.transport = None
            self.engine = None
            return False

    async def stop_engine(self):
        """Clean up the chess engine asynchronously"""
        if self.engine:
            try:
                logger.info("Stopping Stockfish engine")
                await self.engine.quit()
                logger.info("Stockfish engine stopped successfully")
                print("Stockfish engine stopped")
            except Exception as e:
                logger.error(f"Error stopping engine: {e}")
                print(f"Error stopping engine: {e}")
            self.transport = None
            self.engine = None

    async def analyze_position(self, fen: str, depth: int = 3) -> Dict:
        """Analyze a chess position and return evaluation and best moves asynchronously"""
        try:
            logger.debug(f"Analyzing position: {fen[:50]}... (depth={depth})")
            board = chess.Board(fen)
            
            if not self.engine:
                logger.warning("No engine available for analysis")
                return {"evaluation": 0.0, "best_move": None, "pv": []}
            
            info = await self.engine.analyse(board, chess.engine.Limit(depth=depth, time=self.stockfish_time_limit))
            score = info["score"].relative
            
            eval_score = 0.0
            if score.is_mate():
                eval_score = 10000 if score.mate() > 0 else -10000
                logger.debug(f"Mate detected: {score.mate()} moves")
            else:
                eval_score = score.score(mate_score=10000) / 100.0  # Convert to pawns
            
            best_move = info.get("pv", [None])[0]
            pv = info.get("pv", [])
            
            logger.debug(f"Analysis complete: eval={eval_score:.2f}, best_move={best_move}")
            
            return {
                "evaluation": eval_score,
                "best_move": best_move,
                "pv": pv,
                "is_mate": score.is_mate() if hasattr(score, 'is_mate') else False
            }
        except Exception as e:
            logger.error(f"Analysis error for FEN {fen[:30]}: {e}")
            print(f"Analysis error: {e}")
            return {"evaluation": 0.0, "best_move": None, "pv": []}

    def detect_move_type(self, board: chess.Board, move: chess.Move) -> str:
        """Classify the type of move played"""
        if board.is_check():
            return "check"
        elif board.is_capture(move):
            return "capture"
        elif move.promotion:
            return "promotion"
        elif board.is_castling(move):
            return "castling"
        elif len(list(board.legal_moves)) < 10:  # Few legal moves = tactical
            return "tactical"
        else:
            return "positional"

    def evaluate_move_quality(self, prev_eval: float, curr_eval: float, is_white_move: bool) -> str:
        """Evaluate if a move was good, bad, or blunder"""
        # Adjust evaluation based on whose move it was
        eval_change = curr_eval - prev_eval
        if not is_white_move:
            eval_change = -eval_change
        
        if eval_change < -2.0:
            return "blunder"
        elif eval_change < -0.5:
            return "mistake"
        elif eval_change > 0.5:
            return "good"
        else:
            return "normal"

    def generate_opening_commentary(self, board: chess.Board, analysis: Dict) -> str:
        """Generate opening commentary using GPT"""
        if not self.openai_client:
            return "Welcome to this chess match! Let's see how the game unfolds."
        
        try:
            current_eval = analysis.get("evaluation", 0.0)
            
            prompt = f"""
Generate opening commentary for a chess game that's just beginning.

Position: Starting position
Evaluation: {current_eval:.1f}

Create an enthusiastic, welcoming commentary to start the chess match. 
Keep it to 1-2 sentences. Be engaging and set the stage for the game.

Commentary:"""

            response = self.openai_client.chat.completions.create(
                model=self.openai_model,
                messages=[
                    {"role": "system", "content": "You are an entertaining chess commentator. Create engaging opening commentary for chess games."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=min(self.openai_max_tokens, 80),  # Use smaller limit for opening
                temperature=self.openai_temperature
            )
            
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            print(f"Error generating opening commentary: {e}")
            return "Welcome to this chess match! Let's see how the game unfolds."

    def generate_move_commentary(self, board: chess.Board, move: chess.Move, 
                                analysis: Dict, move_quality: str, 
                                previous_board: chess.Board = None) -> str:
        """Generate natural commentary for a specific move using GPT"""
        return self.generate_gpt_commentary(board, move, analysis, move_quality, previous_board)

    async def analyze_game_sequence(self, fen_sequence: List[str], timestamps: List[float]) -> List[CommentaryEvent]:
        """Analyze a sequence of FEN positions and generate commentary events asynchronously"""
        logger.info(f"Starting analysis of {len(fen_sequence)} positions")
        
        if not await self.start_engine():
            logger.warning("Running without engine - commentary will be limited")
            print("Warning: Running without engine - commentary will be limited")
        
        commentary_events = []
        previous_board = None
        
        try:
            for i, (fen, timestamp) in enumerate(zip(fen_sequence, timestamps)):
                try:
                    logger.debug(f"Processing position {i+1}/{len(fen_sequence)} at timestamp {timestamp:.1f}s")
                    board = chess.Board(fen)
                    analysis = await self.analyze_position(fen)
                    current_eval = analysis.get("evaluation", 0.0)
                    
                    # Determine what move was played (if any)
                    move_played = None
                    if previous_board and previous_board.fen() != fen:
                        # Try to find the move that was played
                        for move in previous_board.legal_moves:
                            temp_board = previous_board.copy()
                            temp_board.push(move)
                            if temp_board.fen().split()[0] == fen.split()[0]:  # Compare board position only
                                move_played = move
                                break
                    
                    # Generate commentary
                    if move_played and previous_board:
                        is_white_move = previous_board.turn  # Who just moved
                        move_quality = self.evaluate_move_quality(
                            self.previous_eval, current_eval, is_white_move
                        )
                        
                        move_san = previous_board.san(move_played)
                        logger.debug(f"Found move: {move_san} (quality: {move_quality}, eval change: {current_eval - self.previous_eval:.2f})")
                        
                        commentary_text = self.generate_move_commentary(
                            board, move_played, analysis, move_quality, previous_board
                        )
                        
                        # Determine priority based on move quality and position type
                        priority = 1
                        if move_quality == "blunder":
                            priority = 3
                        elif move_quality in ["good", "tactical"] or abs(current_eval) > 2.0:
                            priority = 2
                        
                        logger.info(f"Generated commentary for {move_san}: {commentary_text[:50]}... (priority: {priority})")
                        
                        event = CommentaryEvent(
                            timestamp=timestamp,
                            text=commentary_text,
                            priority=priority,
                            move_san=move_san,
                            evaluation=current_eval
                        )
                        commentary_events.append(event)
                    
                    elif i == 0:  # Opening comment
                        logger.debug("Generating opening commentary")
                        opening_commentary = self.generate_opening_commentary(board, analysis)
                        logger.info(f"Generated opening commentary: {opening_commentary[:50]}...")
                        event = CommentaryEvent(
                            timestamp=timestamp,
                            text=opening_commentary,
                            priority=1,
                            move_san="",
                            evaluation=current_eval
                        )
                        commentary_events.append(event)
                    
                    self.previous_eval = current_eval
                    previous_board = board
                    self.move_count += 1
                    
                    # Add small delay between analyses to prevent overwhelming the engine
                    await asyncio.sleep(0.1)
                    
                except Exception as e:
                    logger.error(f"Error processing position {i}: {e}")
                    print(f"Error processing position {i}: {e}")
                    continue
        
        finally:
            await self.stop_engine()
        
        logger.info(f"Analysis complete: generated {len(commentary_events)} commentary events")
        return commentary_events

    def save_commentary_script(self, events: List[CommentaryEvent], filename: str):
        """Save commentary events to a JSON file"""
        data = []
        for event in events:
            data.append({
                "timestamp": event.timestamp,
                "text": event.text,
                "priority": event.priority,
                "move_san": event.move_san,
                "evaluation": event.evaluation
            })
        
        with open(filename, 'w') as f:
            json.dump(data, f, indent=2)
        
        print(f"Commentary script saved to {filename}")

    def load_commentary_script(self, filename: str) -> List[CommentaryEvent]:
        """Load commentary events from a JSON file"""
        try:
            with open(filename, 'r') as f:
                data = json.load(f)
            
            events = []
            for item in data:
                event = CommentaryEvent(
                    timestamp=item["timestamp"],
                    text=item["text"],
                    priority=item["priority"],
                    move_san=item.get("move_san", ""),
                    evaluation=item.get("evaluation", 0.0)
                )
                events.append(event)
            
            return events
        except Exception as e:
            print(f"Error loading commentary script: {e}")
            return [] 