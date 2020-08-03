use rayon::prelude::*;
use ndarray::prelude::*;
use ndarray::stack;
use std::sync::Arc;
use std::fmt::Display;
use std::fmt;
use std::sync::Mutex;
use std::thread;
use  std::sync::atomic::{AtomicI32, AtomicUsize, AtomicU8, Ordering, AtomicBool};
use std::time;
use rand::Rng;
use std::ops::Range;
use rand::seq::SliceRandom;
use std::io;

lazy_static! {
    static ref I_PIECE: Arc<Piece> = Arc::new(Piece {
        id: 0,
        bgr: [215, 155, 15],
        orientations: vec![(3, array![[1,1,1,1]]), (5, array![[1], [1], [1], [1]])],
    });

    static ref O_PIECE: Arc<Piece> = Arc::new(Piece {
        id: 1,
        bgr: [2,159,227],
        orientations: vec![(4, array![[1, 1], [1, 1]])],
    });

    static ref T_PIECE: Arc<Piece> = Arc::new(Piece {
        id: 2,
        bgr: [138, 41, 175],
        orientations: vec![(3, array![[0, 1, 0], [1, 1, 1]]), (4, array![[1, 0], [1, 1], [1, 0]]), (3, array![[1, 1, 1], [0, 1, 0]]), (3, array![[0, 1], [1, 1], [0, 1]])],
    });

    static ref S_PIECE: Arc<Piece> = Arc::new(Piece {
        id: 3,
        bgr: [1, 177, 89],
        orientations: vec![(3, array![[0, 1, 1], [1, 1, 0]]), (4, array![[1, 0], [1, 1], [0, 1]])],
    });

    static ref Z_PIECE: Arc<Piece> = Arc::new(Piece {
        id: 4,
        bgr: [55, 15, 215],
        orientations: vec![(3, array![[1, 1, 0], [0, 1, 1]]), (4, array![[0, 1], [1, 1], [1, 0]])],
    });

    static ref J_PIECE: Arc<Piece> = Arc::new(Piece {
        id: 5,
        bgr: [198, 65, 33],
        orientations: vec![(3, array![[1, 0, 0], [1, 1, 1]]), (4, array![[1, 1], [1, 0], [1, 0]]), (3, array![[1, 1, 1], [0, 0, 1]]), (3, array![[0, 1], [0, 1], [1, 1]])],
    });

    static ref L_PIECE: Arc<Piece> = Arc::new(Piece {
        id: 6,
        bgr: [2, 91, 227],
        orientations: vec![(3, array![[0, 0, 1], [1, 1, 1]]), (4, array![[1, 0], [1, 0], [1, 1]]), (3, array![[1, 1, 1], [1, 0, 0]]), (3, array![[1, 1], [0, 1], [0, 1]])],
    });

}

const NUMBER_OF_MOVES: usize = 1000;
const MOVES_PER_THREAD: usize = 20;
const DEPTH: usize = 2;
const NUMBER_OF_TRIALS: usize = 3;
const OPT_RATE: f64 = 1.0;
const TEST_STEP: f64 = 1.0;

fn optimization(range: f64) {
    // Generate random float values for constants
    let mut rng = rand::thread_rng();
    let hole_cost = rng.gen_range(0.0, range);
    let hole_height_cost = rng.gen_range(0.0, range);
    let jagged_cost = rng.gen_range(0.0, range);
    let height_cost = rng.gen_range(0.0, range); 
    let combo_value = rng.gen_range(0.0, range); 
    
    let line_values = [rng.gen_range(0.0, range), rng.gen_range(0.0, range), rng.gen_range(0.0, range), rng.gen_range(0.0, range)]; 

    let height_threshold = rng.gen_range(0, 20);
    
    // Storing Pieces Constants
    let stored_piece_value = [rng.gen_range(-range, range), rng.gen_range(-range, range), rng.gen_range(-range, range), rng.gen_range(-range, range), rng.gen_range(-range, range), rng.gen_range(-range, range), rng.gen_range(-range, range)];

    let constants = OptimizationConstants {
        hole_cost,
        hole_height_cost,
        jagged_cost,
        height_cost,
        combo_value,
        line_values,
        height_threshold,
        stored_piece_value,
    };

    //start a thread that reads the input stream to see if it should stop the program
    thread::spawn(|| {
        loop {
            let mut input = String::new();
            io::stdin().read_line(&mut input).expect("failed to read line");
            if input.trim() == String::from("quit") {
                break;
            }
            else {
                println!("unrecognized request: {}", input.trim());
            }
        }
    });

    let mut handles = Vec::with_capacity(34); 

    for i in 0..17 {
        handles.append(&mut ga_constant(&constants, i));
    }

    // Run Trials w/ Gradient Ascent

    //hole_cost 
    
    run_test_trials(&constants);
}

fn ga_constant (constants: &OptimizationConstants, constant_index: i8) -> Vec<thread::JoinHandle<()>>{
    let mut const_handles = Vec::with_capacity(2);
    let outputs = Arc::new(Mutex::new([0.0, 0.0]));
    let output_clone_pos = Arc::clone(&outputs);
    let output_clone_neg = Arc::clone(&outputs);
    let mut constants_pos;
    let mut constants_neg;
    match constant_index { 
        0 => {
            constants_pos = constants.update_hole_cost(TEST_STEP);
            constants_neg = constants.update_hole_cost(-TEST_STEP);
        }
        1 => {
            constants_pos = constants.update_hole_height_cost(TEST_STEP);
            constants_neg = constants.update_hole_height_cost(-TEST_STEP);
        }
        2 => {
            constants_pos = constants.update_line_values(0, TEST_STEP);
            constants_neg = constants.update_line_values(0, -TEST_STEP);
        }
        3 => {
            constants_pos = constants.update_line_values(1, TEST_STEP);
            constants_neg = constants.update_line_values(1, -TEST_STEP);
        }
        4 => {
            constants_pos = constants.update_line_values(2, TEST_STEP);
            constants_neg = constants.update_line_values(2, -TEST_STEP);
        }
        5 => {
            constants_pos = constants.update_line_values(3, TEST_STEP);
            constants_neg = constants.update_line_values(3, -TEST_STEP);
        }
        6 => {
            constants_pos = constants.update_jagged_cost(TEST_STEP);
            constants_neg = constants.update_jagged_cost(-TEST_STEP);
        }
        7 => {
            constants_pos = constants.update_height_cost(TEST_STEP);
            constants_neg = constants.update_height_cost(-TEST_STEP);
        }
        8 => {
            constants_pos = constants.update_combo_value(TEST_STEP);
            constants_neg = constants.update_combo_value(-TEST_STEP);
        }
        9 => {
            constants_pos = match constants.update_height_threshold(TEST_STEP as u8) {
                Ok(value) => value,
                Err(_) => panic!("Smh idk"),
            };
            constants_neg = match constants.update_height_threshold(-TEST_STEP as u8) {
                Ok(value) => value,
                Err(_) => panic!("Smh idk"),
            };
        }
        10 => {
            constants_pos = constants.update_stored_piece_value(0, TEST_STEP);
            constants_neg = constants.update_stored_piece_value(0, -TEST_STEP);
        }
        11 => {
            constants_pos = constants.update_stored_piece_value(1, TEST_STEP);
            constants_neg = constants.update_stored_piece_value(1, -TEST_STEP);
        }
        12 => {
            constants_pos = constants.update_stored_piece_value(2, TEST_STEP);
            constants_neg = constants.update_stored_piece_value(2, -TEST_STEP);
        }
        13 => {
            constants_pos = constants.update_stored_piece_value(3, TEST_STEP);
            constants_neg = constants.update_stored_piece_value(3, -TEST_STEP);
        }
        14 => {
            constants_pos = constants.update_stored_piece_value(4, TEST_STEP);
            constants_neg = constants.update_stored_piece_value(4, -TEST_STEP);
        }
        15 => {
            constants_pos = constants.update_stored_piece_value(5, TEST_STEP);
            constants_neg = constants.update_stored_piece_value(5, -TEST_STEP);
        }
        16 => {
            constants_pos = constants.update_stored_piece_value(6, TEST_STEP);
            constants_neg = constants.update_stored_piece_value(6, -TEST_STEP);
        }
        _ => {
            panic!("Smh Bruh not a correct index")
        }
    }
    let pos_handle = thread::spawn(move || {
        let tmp = run_test_trials(&constants_pos);
        output_clone_pos.lock().unwrap()[0] = tmp;
    });
    const_handles.push(pos_handle);

    let neg_handle = thread::spawn(move || {
        let tmp = run_test_trials(&constants_neg);
        output_clone_neg.lock().unwrap()[1] = tmp;
    });
    const_handles.push(neg_handle);

    const_handles
}

#[cfg(test)]
mod tests {
    use rand::Rng;
    use super::OptimizationConstants;

    #[test]
    fn epic() {
        let range = 100.0;
        let mut rng = rand::thread_rng();
        let hole_cost = rng.gen_range(0.0, range);
        let hole_height_cost = rng.gen_range(0.0, range);
        let jagged_cost = rng.gen_range(0.0, range);
        let height_cost = rng.gen_range(0.0, range); 
        let combo_value = rng.gen_range(0.0, range); 
        
        let line_values = [rng.gen_range(0.0, range), rng.gen_range(0.0, range), rng.gen_range(0.0, range), rng.gen_range(0.0, range)]; 

        let height_threshold = rng.gen_range(0, 20);
        
        // Storing Pieces Constants
        let stored_piece_value = [rng.gen_range(-range, range), rng.gen_range(-range, range), rng.gen_range(-range, range), rng.gen_range(-range, range), rng.gen_range(-range, range), rng.gen_range(-range, range), rng.gen_range(-range, range)];

        let constants = OptimizationConstants {
            hole_cost,
            hole_height_cost,
            jagged_cost,
            height_cost,
            combo_value,
            line_values,
            height_threshold,
            stored_piece_value,
        };
        super::optimization(100.0);
    }
}

fn run_test_trials(constants: &OptimizationConstants) -> f64 {
    let mut average_attack = 0.0;
    let mut trial_handles = Vec::with_capacity(NUMBER_OF_TRIALS);
    for i in 0..NUMBER_OF_TRIALS {
        let tmp_constants = constants.clone();
        let handle = thread::spawn(move || {
            match run_test(NUMBER_OF_MOVES, tmp_constants.hole_cost, tmp_constants.hole_height_cost, tmp_constants.line_values, tmp_constants.jagged_cost, tmp_constants.height_cost, tmp_constants.combo_value, tmp_constants.height_threshold, tmp_constants.stored_piece_value) {
                TestResult::Lost(num_moves, attack) => {
                    let adjusted_attack = attack * num_moves / NUMBER_OF_MOVES as i32;
                    println!("Meme Lost Smh: {}", num_moves);
                    println!("Attack");
                    println!("Adjusted attack: {}", adjusted_attack);
                    average_attack += adjusted_attack as f64;
                },
                TestResult::Complete(attack) => {
                    println!("Finished with attack: {}", attack);
                    average_attack += attack as f64;
                },
            };
        });
        trial_handles.push(handle);
    }
    for handle in trial_handles {
        handle.join().unwrap();
    }
    average_attack = average_attack as f64/NUMBER_OF_TRIALS as f64;
    average_attack
}

fn run_test(num_moves: usize, hole_cost: f64, hole_height_cost: f64, line_values: [f64;4], jagged_cost: f64, height_cost: f64, combo_value: f64, height_threshold: u8, stored_piece_value: [f64;7]) -> TestResult {
    let mut generator = PieceGenerator::new();
    let mut field = Field::new([generator.get_next_piece(), generator.get_next_piece(), generator.get_next_piece(), generator.get_next_piece(), generator.get_next_piece()], hole_cost, hole_height_cost, line_values, jagged_cost, height_cost, combo_value, height_threshold, stored_piece_value);
    
    for i in 0..num_moves {
        let suggested_move = match field.calculate_all_resulting_fields_new(DEPTH, true) {
            Some(m) => m,
            None => panic!("smh at the disco"),
        };
        field = match Field::from(&field, &suggested_move, &Piece::get_piece_from_id(field.upcoming_pieces[0]).unwrap()) {
            Ok(m) => m,
            Err(_) => {
                return TestResult::Lost(i as i32, field.total_attack);
            },
        };
        let mut new_upcoming = [0;5];
        for i in 0..4 { 
            new_upcoming[i] = field.upcoming_pieces[i + 1];
        }
        field.upcoming_pieces = new_upcoming;
        field.upcoming_pieces[4] = generator.get_next_piece();
        field.value = 0.0;
        //println!("field: {}, move: {}", field, i);
    }
    TestResult::Complete(field.total_attack)
}

enum TestResult {
    Lost(i32, i32),
    Complete(i32),
}
 
#[derive(Clone)]
struct OptimizationConstants {
    hole_cost: f64, 
    hole_height_cost: f64, 
    line_values: [f64;4], 
    jagged_cost: f64, 
    height_cost: f64, 
    combo_value: f64, 
    height_threshold: u8, 
    stored_piece_value: [f64;7],
}

impl OptimizationConstants {
    fn new(hole_cost: f64, hole_height_cost: f64, line_values: [f64; 4], jagged_cost: f64, height_cost: f64, combo_value: f64, height_threshold: u8, stored_piece_value: [f64; 7]) -> OptimizationConstants {
        OptimizationConstants {
            hole_cost,
            hole_height_cost,
            line_values,
            jagged_cost,
            height_cost, 
            combo_value, 
            height_threshold,
            stored_piece_value,
        }
    }
    fn update_hole_cost(&self, change: f64) -> OptimizationConstants {
        let mut changed_constants = self.clone();
        changed_constants.hole_cost += change;
        changed_constants
    }
    fn update_hole_height_cost(&self, change: f64) -> OptimizationConstants {
        let mut changed_constants = self.clone();
        changed_constants.hole_height_cost += change;
        changed_constants
    }
    fn update_line_values(&self, index: usize, change: f64) -> OptimizationConstants {
        let mut changed_constants = self.clone();
        changed_constants.line_values[index] += change;
        changed_constants
    }
    fn update_jagged_cost(&self, change: f64) -> OptimizationConstants {
        let mut changed_constants = self.clone();
        changed_constants.jagged_cost += change;
        changed_constants
    }
    fn update_height_cost(&self, change: f64) -> OptimizationConstants {
        let mut changed_constants = self.clone();
        changed_constants.height_cost += change;
        changed_constants
    }
    fn update_combo_value(&self, change: f64) -> OptimizationConstants {
        let mut changed_constants = self.clone();
        changed_constants.combo_value += change;
        changed_constants
    }
    fn update_height_threshold(&self, change: u8) -> Result<OptimizationConstants, &str> {
        let mut changed_constants = self.clone();
        if (changed_constants.height_threshold as i8 + change as i8) < 0 {
            return Err("smh new height is not U8");
        }
        changed_constants.height_threshold += change;
        Ok(changed_constants)
    }
    fn update_stored_piece_value(&self, index: usize, change: f64) -> OptimizationConstants {
        let mut changed_constants = self.clone();
        changed_constants.stored_piece_value[index] += change;
        changed_constants
    }
}

#[derive(Clone)]
struct PieceGenerator {
    upcoming_pieces: Vec<u8>,
}

impl PieceGenerator {
    fn new() -> PieceGenerator {
        PieceGenerator {
            upcoming_pieces: PieceGenerator::get_next_bag(),
        }
    }
    fn get_next_piece(&mut self) -> u8 {
        if self.upcoming_pieces.len() == 0 {
            self.upcoming_pieces = PieceGenerator::get_next_bag();
        }
        self.upcoming_pieces.remove(0)
    }
    fn get_next_bag() -> Vec<u8> {
        let mut rng = rand::thread_rng();
        let mut upcoming_piece_bag = [0, 1, 2, 3, 4, 5, 6];
        upcoming_piece_bag.shuffle(&mut rng);
        Vec::from(upcoming_piece_bag)
    }

}


#[derive(Clone)]
struct Field {
    field_state: Array::<u8, Ix2>,
    upcoming_pieces: [u8;5],
    stored_piece: Option<u8>,
    value: f64,
    combo: usize,
    back_to_back: bool,
    
    total_attack: i32,
    
    constants: OptimizationConstants,
}

impl Field {
    fn new(upcoming_pieces: [u8;5], hole_cost: f64, hole_height_cost: f64, line_values: [f64; 4], jagged_cost: f64, height_cost: f64, combo_value: f64, height_threshold: u8, stored_piece_value: [f64; 7]) -> Field {
        Field {
            field_state: Array::<u8, Ix2>::zeros((20, 10)),
            // Create a three-dimensional f64 array, initialized with zeros
            upcoming_pieces,
            stored_piece: None,
            value: 0.0,
            combo: 0,
            back_to_back: false,

            total_attack: 0,
            
            constants: OptimizationConstants {
                hole_cost,
                hole_height_cost,
                line_values,
                jagged_cost,
                height_cost,
                combo_value,
                height_threshold,
                stored_piece_value,
            },
        }
    }
    fn from<'a>(prior_field: &Field, new_move: &Move, new_piece: &Piece) -> Result<Field, &'a str> {
        let tmp2;
        let mut new_piece = new_piece;
        let stored_piece = if new_move.store == true {
            match prior_field.stored_piece {
                Some(p) => {
                    let tmp = new_piece.id;
                    tmp2 = Piece::get_piece_from_id(p).unwrap().clone();
                    new_piece = &tmp2;
                    Some(tmp)
                }
                None => panic!("Smh. we are lazy and don't want to deal with a situation where there isn't a piece already stored xdddddd"),
            }
        } else {
            prior_field.stored_piece.clone()
        };
        let mut f = Field {
            //switches the stored piece if the move is "true" for store
            stored_piece,
             //But meme maybe change later
            
            field_state: {
                let mut field_state = prior_field.field_state.clone();
                let x_start = (new_move.position + new_piece.orientations[new_move.rotation as usize].0) as usize;
                let piece_width = new_piece.orientations[new_move.rotation as usize].1.shape()[1];
                let current_field = field_state.slice(s![0..20,  x_start..x_start + piece_width]);
                //FOR LOOP METHOD
                let mut max_block_y_value = 20;
                for y in 0..20 {
                    match current_field.slice(s![y, ..]).into_iter().find(|&&x| x == 1) {
                        Some(v) => {
                            max_block_y_value = y;
                            break;
                        },
                        None => continue,
                    };
                }
        
                //ITER METHOD
                /*let mut max_block_y_value = match current_field.into_iter().position(|&v| v == 1) {
                    Some(index) => (index - index%piece_width)/piece_width,
                    None => 20,
                };*/
        
                let mut danger_height = false;
                max_block_y_value = if max_block_y_value < new_piece.orientations[new_move.rotation as usize].1.shape()[0] {
                    danger_height = true;
                    new_piece.orientations[new_move.rotation as usize].1.shape()[0]
                }
                else {
                    max_block_y_value
                };
                let mut found = false;
                let piece_size = new_piece.orientations[new_move.rotation as usize].1.shape();
                for y in max_block_y_value..20 + piece_size[0] {
                    if y > 20 {
                        break;
                    }
                    let test_field_section = &current_field.slice(s![y - piece_size[0]..y, ..]) + &new_piece.orientations[new_move.rotation as usize].1;
                    match test_field_section.into_iter().find(|&&x| x == 2) {
                        Some(_v) => {
                            if danger_height {
                                return Err("smh too high");
                            }
                            let mut tmp = field_state.slice_mut(s![y - 1 - piece_size[0]..y - 1, x_start..x_start + new_piece.orientations[new_move.rotation as usize].1.shape()[1]]);
                            tmp += &new_piece.orientations[new_move.rotation as usize].1;
                            found = true;
                            break;
                        },
                        None => {
                            danger_height = false;
                            continue;
                        },
                    }
                }
                if !found {
                    let mut tmp = field_state.slice_mut(s![20 - piece_size[0]..20, x_start..x_start + new_piece.orientations[new_move.rotation as usize].1.shape()[1]]);
                    tmp += &new_piece.orientations[new_move.rotation as usize].1;
                }
                field_state
            },
            upcoming_pieces: prior_field.upcoming_pieces, 
            

            value: prior_field.value,
            combo: prior_field.combo,
            back_to_back: prior_field.back_to_back,

            total_attack: prior_field.total_attack,
            
            constants: prior_field.constants.clone(),
            
        };
        Field::clean_sent_lines(&mut f);
        Ok(f)
    }
    fn eval_sent_lines(attack:usize, field: &mut Field) -> f64 {
        match attack {
            0 => 0.0, 
            1 => {
                field.back_to_back = false;
                field.total_attack += 0;
                field.constants.line_values[0]
            },
            2 => {
                field.back_to_back = false;
                field.total_attack += 1;
                field.constants.line_values[1]
            },
            3 => {
                field.back_to_back = false;
                field.total_attack += 2;
                field.constants.line_values[2]
            },
            4 => {
                field.total_attack += 4;
                if field.back_to_back {
                    field.total_attack += 1;
                }
                else {
                    field.back_to_back = true;
                }
                field.constants.line_values[3]
            },
            _ => {
                field.back_to_back = false;
                field.total_attack = attack as i32;
                100000.0
            },
        }
    }
    fn eval_combo(&mut self, combo:usize) {
        match combo {
            0 => return, 
            1 => return,
            1 => {
                self.total_attack += 0;
            },
            2..=4 => {
                self.total_attack += 1;
            },
            5 | 6 => {
                self.total_attack += 2;
            },
            7 | 8 => {
                self.total_attack += 3;
            },
            9..=11 => {
                self.total_attack += 4;
            },
            _ => {
                self.total_attack += 5;
            },
        }
    }
    fn clean_sent_lines(field: &mut Field){
        let mut count = 0;
        let blank = Array::<u8, Ix2>::zeros((1,10));
        for y in 0..20 {
            if field.field_state.slice(s![y, 0..10]).sum() == 10 {
                count += 1;
                field.field_state = stack![Axis(0), blank.view(), field.field_state.slice(s![0..y, 0..10]), field.field_state.slice(s![y + 1..20, 0..10])];
            }
        }
        if (field.field_state.slice(s![19, 0..10])).sum() == 0 {
            count = 10;
        }
        if count != 0 {
            field.eval_combo(field.combo);
            field.combo += 1;
        }
        else {
            field.value += field.combo as f64 * field.constants.combo_value;
            field.combo = 0;
        }
        field.value += Field::eval_sent_lines(count, field);
    }
    fn eval(&self) -> f64 {
        let mut y_values = [0; 10];
        let mut height_costs = 0;
        let mut hole_count = 0;
        let mut hole_value = 0;
        let state = &self.field_state;
        let mut jaggedness = 0;
        for x in 0..10 {
            let mut height_found = false;
            //Heights
            match state.slice(s![.., x]).iter().position(|&y| y == 1) {
                Some(index) => {
                    height_found = true;
                    //height_sum += 20 - index;
                    y_values[x] = index;
                    if (y_values[x] as u8) < (20 - &self.constants.height_threshold) {
                        height_costs += 20 - y_values[x] as u8 - self.constants.height_threshold;
                    }
                }
                None => {
                    y_values[x] = 20;
                    continue
                }
            }
            if x > 0 {
                jaggedness += (y_values[x] as isize - y_values[x-1] as isize).pow(2);
            }
            //Holes
            if height_found {
                for y in y_values[x]..20 {
                    if state[[y, x]] == 0 {
                        hole_value += y - y_values[x];
                        hole_count += 1;
                    }
                }
            }
        }
        let score = -(self.constants.jagged_cost * jaggedness as f64 + self.constants.hole_height_cost * hole_value as f64 + self.constants.hole_cost * hole_count as f64 + self.constants.height_cost * height_costs as f64);
        score
    }
    
    fn calculate_all_resulting_fields(&self, depth: usize) -> Option<Move> {
        if depth == 0 {
            println!("{}",self.eval());
            None
        }
        else {
            let child_result = Arc::new(Mutex::new((0, f64::MIN)));
            let piece = Piece::get_piece_from_id(self.upcoming_pieces[0]).unwrap();
            let mut handles = Vec::new();
            let mut moves = if let Some(stored_piece) = self.stored_piece {
                let mut vec = piece.get_all_moves(false);
                vec.append(&mut Piece::get_piece_from_id(stored_piece).unwrap().get_all_moves(true));
                vec
            }
            else {
                piece.get_all_moves(false)
            };
            let original_depth = depth;
            let start = time::Instant::now();
            for i in 0..moves.len() {
                let field = match Field::from(self, &moves[i], &piece) {
                    Ok(f) => f,
                    Err(_e) => continue,
                };
                let result_clone = Arc::clone(&child_result);
                let handle = thread::spawn(move || {
                    recursive_resulting_fields(field, depth - 1, i, result_clone, original_depth);
                });
                handles.push(handle);
            }  
            //println!("initial thread creation time: {:?}", start.elapsed());
            let start = time::Instant::now();  
            for handle in handles {
                handle.join().expect("heck I died lol");
            }
            //println!("finished: {:?}", start.elapsed());
            let move_id = child_result.lock().unwrap().0;
            Some(moves[move_id])
        }
    }
    fn calculate_all_resulting_fields_new(&self, depth: usize, stored: bool) -> Option<Move> {
        if depth == 0 {
            println!("{}",self.eval());
            None
        }
        else {
            let child_result = Arc::new(Mutex::new((0, f64::MIN)));
            let piece = Piece::get_piece_from_id(self.upcoming_pieces[0]).unwrap();
            let mut handles = Vec::new();
            let moves = if stored == true {if let Some(stored_piece) = self.stored_piece {
                let mut vec = piece.get_all_moves(false);
                vec.append(&mut Piece::get_piece_from_id(stored_piece).unwrap().get_all_moves(true));
                vec
            } else {
                piece.get_all_moves(false)
            }} else {
                piece.get_all_moves(false)
            };
            
            let original_depth = depth;
            let overlap = moves.len() % MOVES_PER_THREAD;
            let total_iterations = moves.len()/MOVES_PER_THREAD;
            if total_iterations > 0 {
                let mut moves_clone = moves.clone();
                let extra = overlap/total_iterations;
                let mut extra_extra = overlap % total_iterations;
                //println!("overlap: {}", overlap);
                //println!("len:{}", moves_clone.len());
                let start = time::Instant::now(); 
                for i in 0..total_iterations {
                    let result_clone = Arc::clone(&child_result);
                    //sets the number of moves that will be passed into the thread created for the current iteration
                    let length = if extra_extra == 0 {
                        MOVES_PER_THREAD + extra
                    } else {
                        extra_extra -= 1;
                        MOVES_PER_THREAD + extra + 1
                    };

                    

                    let field = self.clone();

                    //for keeping track of move ids
                    let moves_start_index = moves_clone.len() - length;
                    
                    //moves that will be passed into the thread for this iteration
                    let tmp_moves = moves_clone.split_off(moves_start_index);
                    //println!("len:{}", moves_clone.len());
                    //overall mutex clone
                    let result_clone = Arc::clone(&child_result);

                    let piece = piece.clone();

                    //spawn new thread
                    let handle = thread::spawn(move || {
                        let child_result_thread = Arc::new(Mutex::new((0, f64::MIN)));
                        let mut child_handles = Vec::with_capacity(length);
                        for i in 0..length {
                            let child_result_clone = Arc::clone(&child_result_thread);
                            let new_field = match Field::from(&field, &tmp_moves[i], &piece) {
                                Ok(f) => f,
                                Err(_) => continue,
                            };
                            let interior_handle = thread::spawn(move || {
                                recursive_resulting_fields_new(new_field, depth - 1, moves_start_index + i, child_result_clone, original_depth)
                            });
                            child_handles.push(interior_handle);
                        }
                        for handle in child_handles {
                            handle.join().unwrap();
                        }
                        let best_move = *child_result_thread.lock().unwrap();
                        let mut max_eval = result_clone.lock().unwrap();
                        if max_eval.1 < best_move.1 {
                            *max_eval = best_move;
                        }
                    });
                    handles.push(handle);
                }
            }
            else {
                for i in 0..moves.len() {
                    let field = match Field::from(self, &moves[i], &piece) {
                        Ok(f) => f,
                        Err(_e) => continue,
                    };
                    let result_clone = Arc::clone(&child_result);
                    let handle = thread::spawn(move || {
                        recursive_resulting_fields(field, depth - 1, i, result_clone, original_depth);
                    });
                    handles.push(handle);
                } 
            }
            
            //println!("initial thread creation time: {:?}", start.elapsed());
            let start = time::Instant::now();  
            for handle in handles {
                handle.join().expect("heck I died lol");
            }
            //println!("finished: {:?}", start.elapsed());
            let move_id = child_result.lock().unwrap().0;
            Some(moves[move_id])
        }
    }
}
fn recursive_resulting_fields(f: Field, depth: usize, move_id: usize, result: Arc<Mutex<(usize, f64)>>, original_depth: usize) {
    if depth == 0 {
        let eval = f.eval();
        let mut max_eval = result.lock().unwrap();
        //println!("{},{}",move_id, eval);
        //println!("{}", f);
        if max_eval.1 < eval {
            *max_eval = (move_id, eval);
            //println!("{},{}",move_id, eval);
        }
    }
    else {
        let start = time::Instant::now();
        let child_result = Arc::new(Mutex::new((0, f64::MIN)));
        let piece = Piece::get_piece_from_id(f.upcoming_pieces[original_depth - depth]).unwrap();
        let mut handles = Vec::new();
        for m in if let Some(stored_piece) = f.stored_piece {
            let mut vec = piece.get_all_moves(false);
            vec.append(&mut Piece::get_piece_from_id(stored_piece).unwrap().get_all_moves(true));
            vec
        }
        else {
            piece.get_all_moves(false)
        } {
            let field = match Field::from(&f, &m, &piece) {
                Ok(f) => f,
                Err(_e) => continue,
            };
            let result_clone = Arc::clone(&child_result);
            let handle = thread::spawn(move || {
                recursive_resulting_fields(field, depth - 1, move_id, result_clone, original_depth);
            });
            handles.push(handle);
        }
        //println!("thread creation time: {:?}", start.elapsed());    
        for handle in handles {
            handle.join().unwrap();
        }
        let mut max_eval = result.lock().unwrap();
        let child_result = child_result.lock().unwrap().1;
        if max_eval.1 < child_result {
            *max_eval = (move_id, child_result);
        }
    }
}

fn recursive_resulting_fields_new(f: Field, depth: usize, move_id: usize, result: Arc<Mutex<(usize, f64)>>, original_depth: usize) {
    if depth == 0 {
        let eval = f.eval();
        let mut max_eval = result.lock().unwrap();
        //println!("{},{}",move_id, eval);
        //println!("{}", f);
        if max_eval.1 < eval {
            *max_eval = (move_id, eval);
            //println!("{},{}",move_id, eval);
        }
    }
    else {
        let start = time::Instant::now();
        let child_result = Arc::new(Mutex::new((0, f64::MIN)));
        let piece = Piece::get_piece_from_id(f.upcoming_pieces[original_depth - depth]).unwrap();

        let moves = if let Some(stored_piece) = f.stored_piece {
            let mut vec = piece.get_all_moves(false);
            vec.append(&mut Piece::get_piece_from_id(stored_piece).unwrap().get_all_moves(true));
            vec
        }
        else {
            piece.get_all_moves(false)
        };
        let mut handles = Vec::new();
        
        let mut overlap = moves.len() % MOVES_PER_THREAD;
        let total_iterations = moves.len()/MOVES_PER_THREAD;
        if total_iterations > 0 {
            let extra = overlap/total_iterations;
            let mut extra_extra = overlap % total_iterations;
            let mut moves_clone = moves.clone();

            
            for i in 0..total_iterations {
                let result_clone = Arc::clone(&child_result);
                //sets the number of moves that will be passed into the thread created for the current iteration
                let length = if extra_extra == 0 {
                    MOVES_PER_THREAD + extra
                } else {
                    extra_extra -= 1;
                    MOVES_PER_THREAD + extra + 1
                };

                let field = f.clone();

                //for keeping track of move ids
                let moves_start_index = moves_clone.len() - length;
                
                //moves that will be passed into the thread for this iteration
                let tmp_moves = moves_clone.split_off(moves_start_index);

                //overall mutex clone
                let result_clone = Arc::clone(&child_result);

                let piece = piece.clone();

                //spawn new thread
                let handle = thread::spawn(move || {
                    let child_result_thread = Arc::new(Mutex::new((0, f64::MIN)));
                    let mut child_handles = Vec::with_capacity(length);
                    for i in 0..length {
                        let child_result_clone = Arc::clone(&child_result_thread);
                        let new_field = match Field::from(&field, &tmp_moves[i], &piece) {
                            Ok(f) => f,
                            Err(_) => continue,
                        };
                        let interior_handle = thread::spawn(move || {
                            recursive_resulting_fields_new(new_field, depth - 1, moves_start_index + i, child_result_clone, original_depth)
                        });
                        child_handles.push(interior_handle);
                    }
                    for handle in child_handles {
                        handle.join().unwrap();
                    }
                    let best_move = *child_result_thread.lock().unwrap();
                    let mut max_eval = result_clone.lock().unwrap();
                    if max_eval.1 < best_move.1 {
                        *max_eval = best_move;
                    }
                });
                handles.push(handle);
            }
        }
        else {
            for m in moves {
                let field = match Field::from(&f, &m, &piece) {
                    Ok(f) => f,
                    Err(_e) => continue,
                };
                let result_clone = Arc::clone(&child_result);
                let handle = thread::spawn(move || {
                    recursive_resulting_fields(field, depth - 1, move_id, result_clone, original_depth);
                });
                handles.push(handle);
            }
        }
        //println!("thread creation time: {:?}", start.elapsed());    
        for handle in handles {
            handle.join().unwrap();
        }
        let mut max_eval = result.lock().unwrap();
        let child_result = child_result.lock().unwrap().1;
        if max_eval.1 < child_result {
            *max_eval = (move_id, child_result);
        }
    }
}

impl Display for Field {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let mut output = String::new();
        for y in 0..20 {
            for x in 0..10 {
                output = format!("{} {}", output, self.field_state[[y,x]]);
            }
            output = format!("{}{}", output, "\n");
        }
        write!(f, "({}\nUpcoming Pieces: {:?}, Stored: {})", output, self.upcoming_pieces, match self.stored_piece {Some(p) => p.to_string(), None => String::from("N")})
    }
}

#[derive(Copy, Clone)]
struct Move {
    rotation: u8,
    position: i8,
    store: bool,
}

impl Move {
    fn from(m: (u8, i8, bool)) -> Move {
        Move {
            rotation: m.0,
            position: m.1,
            store: m.2,
        }
    }
}

impl Display for Move {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "(Rotation: {}, Position: {})", self.rotation, self.position)
    }
}

#[derive(Clone)]
struct Piece {
    id: u8,
    bgr: [u8;3],
    //Orientation: how many blocks from left, matrix representing dimensions
    orientations: Vec<(i8, Array2::<u8>)>,
}

impl Piece {
    fn get_all_moves(&self, store: bool) -> Vec<Move> {
        let mut moves = Vec::new();
        for i in 0..self.orientations.len() {
            for x in -self.orientations[i].0..(10 - self.orientations[i].1.shape()[1] as i8 - self.orientations[i].0 + 1) {
                moves.push(Move {
                    rotation: i as u8,
                    position: x,
                    store,
                })
            }
        }
        moves
    }
    fn get_piece_from_id<'a>(id: u8) -> Result<Arc<Piece>, &'a str> {
        match id {
            0 => Ok(Arc::clone(&I_PIECE)),
            1 => Ok(Arc::clone(&O_PIECE)),
            2 => Ok(Arc::clone(&T_PIECE)),
            3 => Ok(Arc::clone(&S_PIECE)),
            4 => Ok(Arc::clone(&Z_PIECE)),
            5 => Ok(Arc::clone(&J_PIECE)),
            6 => Ok(Arc::clone(&L_PIECE)),
            _ => Err("Specified piece id does not exist!"),
        }
    }
}
