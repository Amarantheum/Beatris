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
use std::fs;
use std::fs::File;
use std::path::{Path, PathBuf};
use std::fs::OpenOptions;
use std::io::Write;
use std::io::BufReader;
use std::io::BufRead;
use std::io::{Error, ErrorKind};
use num_cpus;


lazy_static! {
    static ref I_PIECE: Arc<Piece> = Arc::new(Piece {
        id: 0,
        bgr: [215, 155, 15],
        orientations: vec![(3, array![[1,1,1,1]]), (5, array![[1], [1], [1], [1]])],
        possible_moves_true: Piece::get_all_moves_prev(vec![(3, array![[1,1,1,1]]), (5, array![[1], [1], [1], [1]])], true),
        possible_moves_false: Piece::get_all_moves_prev(vec![(3, array![[1,1,1,1]]), (5, array![[1], [1], [1], [1]])], false),
    });

    static ref O_PIECE: Arc<Piece> = Arc::new(Piece {
        id: 1,
        bgr: [2,159,227],
        orientations: vec![(4, array![[1, 1], [1, 1]])],
        possible_moves_true: Piece::get_all_moves_prev(vec![(4, array![[1, 1], [1, 1]])], true),
        possible_moves_false: Piece::get_all_moves_prev(vec![(4, array![[1, 1], [1, 1]])], false),
    });

    static ref T_PIECE: Arc<Piece> = Arc::new(Piece {
        id: 2,
        bgr: [138, 41, 175],
        orientations: vec![(3, array![[0, 1, 0], [1, 1, 1]]), (4, array![[1, 0], [1, 1], [1, 0]]), (3, array![[1, 1, 1], [0, 1, 0]]), (3, array![[0, 1], [1, 1], [0, 1]])],
        possible_moves_true: Piece::get_all_moves_prev(vec![(3, array![[0, 1, 0], [1, 1, 1]]), (4, array![[1, 0], [1, 1], [1, 0]]), (3, array![[1, 1, 1], [0, 1, 0]]), (3, array![[0, 1], [1, 1], [0, 1]])], true),
        possible_moves_false: Piece::get_all_moves_prev(vec![(3, array![[0, 1, 0], [1, 1, 1]]), (4, array![[1, 0], [1, 1], [1, 0]]), (3, array![[1, 1, 1], [0, 1, 0]]), (3, array![[0, 1], [1, 1], [0, 1]])], false),
    });

    static ref S_PIECE: Arc<Piece> = Arc::new(Piece {
        id: 3,
        bgr: [1, 177, 89],
        orientations: vec![(3, array![[0, 1, 1], [1, 1, 0]]), (4, array![[1, 0], [1, 1], [0, 1]])],
        possible_moves_true: Piece::get_all_moves_prev(vec![(3, array![[0, 1, 1], [1, 1, 0]]), (4, array![[1, 0], [1, 1], [0, 1]])], true),
        possible_moves_false: Piece::get_all_moves_prev(vec![(3, array![[0, 1, 1], [1, 1, 0]]), (4, array![[1, 0], [1, 1], [0, 1]])], false),
    });

    static ref Z_PIECE: Arc<Piece> = Arc::new(Piece {
        id: 4,
        bgr: [55, 15, 215],
        orientations: vec![(3, array![[1, 1, 0], [0, 1, 1]]), (4, array![[0, 1], [1, 1], [1, 0]])],
        possible_moves_true: Piece::get_all_moves_prev(vec![(3, array![[1, 1, 0], [0, 1, 1]]), (4, array![[0, 1], [1, 1], [1, 0]])], true),
        possible_moves_false: Piece::get_all_moves_prev(vec![(3, array![[1, 1, 0], [0, 1, 1]]), (4, array![[0, 1], [1, 1], [1, 0]])], false),
    });

    static ref J_PIECE: Arc<Piece> = Arc::new(Piece {
        id: 5,
        bgr: [198, 65, 33],
        orientations: vec![(3, array![[1, 0, 0], [1, 1, 1]]), (4, array![[1, 1], [1, 0], [1, 0]]), (3, array![[1, 1, 1], [0, 0, 1]]), (3, array![[0, 1], [0, 1], [1, 1]])],
        possible_moves_true: Piece::get_all_moves_prev(vec![(3, array![[1, 0, 0], [1, 1, 1]]), (4, array![[1, 1], [1, 0], [1, 0]]), (3, array![[1, 1, 1], [0, 0, 1]]), (3, array![[0, 1], [0, 1], [1, 1]])], true),
        possible_moves_false: Piece::get_all_moves_prev(vec![(3, array![[1, 0, 0], [1, 1, 1]]), (4, array![[1, 1], [1, 0], [1, 0]]), (3, array![[1, 1, 1], [0, 0, 1]]), (3, array![[0, 1], [0, 1], [1, 1]])], false),
    });

    static ref L_PIECE: Arc<Piece> = Arc::new(Piece {
        id: 6,
        bgr: [2, 91, 227],
        orientations: vec![(3, array![[0, 0, 1], [1, 1, 1]]), (4, array![[1, 0], [1, 0], [1, 1]]), (3, array![[1, 1, 1], [1, 0, 0]]), (3, array![[1, 1], [0, 1], [0, 1]])],
        possible_moves_true: Piece::get_all_moves_prev(vec![(3, array![[0, 0, 1], [1, 1, 1]]), (4, array![[1, 0], [1, 0], [1, 1]]), (3, array![[1, 1, 1], [1, 0, 0]]), (3, array![[1, 1], [0, 1], [0, 1]])], true),
        possible_moves_false: Piece::get_all_moves_prev(vec![(3, array![[0, 0, 1], [1, 1, 1]]), (4, array![[1, 0], [1, 0], [1, 1]]), (3, array![[1, 1, 1], [1, 0, 0]]), (3, array![[1, 1], [0, 1], [0, 1]])], false),
    });

    static ref STOP: Mutex<bool> = Mutex::new(false);

    static ref NUM_CPUS: usize = num_cpus::get();

    static ref TP: rayon::ThreadPool = rayon::ThreadPoolBuilder::new().num_threads(num_cpus::get()).build().unwrap();
}

const NUMBER_OF_MOVES: usize = 10;
const MOVES_PER_THREAD: usize = 20;
const DEPTH: usize = 2;
const NUMBER_OF_TRIALS: usize = 3;
const OPT_RATE: f64 = 100.0;
const TEST_STEP: f64 = 1.0;
const STORED_PIECES: bool = false;


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
    let stored_piece_value = if !STORED_PIECES {
        None
    } else {
        Some([rng.gen_range(-range, range), rng.gen_range(-range, range), rng.gen_range(-range, range), rng.gen_range(-range, range), rng.gen_range(-range, range), rng.gen_range(-range, range), rng.gen_range(-range, range)])
    };

    let mut constants = OptimizationConstants {
        hole_cost,
        hole_height_cost,
        jagged_cost,
        height_cost,
        combo_value,
        line_values,
        height_threshold,
        stored_piece_value,

        value: None,
    };

    let mut count = 0;
    loop {
        if constants.value == None {
            constants.value = Some(run_test_trials(&constants));
            OptimizationConstants::save_to_file(&constants).unwrap();
        }

        //start a thread that reads the input stream to see if it should stop the program
        /*thread::spawn(|| {
            loop {
                let mut input = String::new();
                io::stdin().read_line(&mut input).expect("failed to read line");
                if input.trim() == String::from("quit") {
                    *STOP.lock().unwrap() = true;
                    break;
                }
                else {
                    println!("unrecognized request: {}", input.trim());
                }
            }
        });*/

        let num_constants = if STORED_PIECES {
            17
        } else {
            10
        };

        let mut mutexes = Vec::with_capacity(num_constants);
        
        TP.install(|| {
            rayon::scope(|s| {
                for i in 0..num_constants {
                    match ga_constant(&constants, i as i8, s) {
                        Ok(vec) => {
                            mutexes.push(Some(vec));
                        },
                        Err(_) => {
                            mutexes.push(None);
                            continue;
                        },
                    }
                }
            });
        });
        

        let mut dy_list = Vec::with_capacity(num_constants);

        for mutex in &mutexes {
            if let Some(m) = mutex {
                let values = m.lock().unwrap();
                dy_list.push(Some(values[0] - values[1]));
            }
            else {
                dy_list.push(None);
            }
        }
        
        println!("{:?}", dy_list);

        let mut final_constants = constants.clone();
        if let Some(dy) = dy_list[0] {
            final_constants = final_constants.update_hole_cost(dy * OPT_RATE);
        }
        else {
            panic!("what the nani why is dy_list None for hole_cost?!");
        }

        if let Some(dy) = dy_list[1] {
            final_constants = final_constants.update_hole_height_cost(dy * OPT_RATE);
        }
        else {
            panic!("what the nani why is dy_list None for hole_height_cost?!");
        }

        for i in 0..4 {
            if let Some(dy) = dy_list[i + 2] {
                final_constants = final_constants.update_line_values(i, dy * OPT_RATE);
            }
            else {
                panic!("what the nani why is dy_list None for line_values?!");
            }
        }
        
        if let Some(dy) = dy_list[6] {
            final_constants = final_constants.update_jagged_cost(dy * OPT_RATE);
        }
        else {
            panic!("what the nani why is dy_list None for jagged_cost?!");
        }

        if let Some(dy) = dy_list[7] {
            final_constants = final_constants.update_height_cost(dy * OPT_RATE);
        }
        else {
            panic!("what the nani why is dy_list None for height_cost?!");
        }

        if let Some(dy) = dy_list[8] {
            final_constants = final_constants.update_combo_value(dy * OPT_RATE);
        }
        else {
            panic!("what the nani why is dy_list None for combo_value?!");
        }

        if let Some(dy) = dy_list[9] {
            final_constants = match final_constants.update_height_threshold((dy/dy.abs()) as i8) {
                Ok(_) => {
                    let m = if let Some(m) = &mutexes[9] {
                        let x = m.lock().unwrap().clone();
                        x
                    }
                    else {
                        panic!("The mutexes for height threshold are trolling.")
                    };

                    let normal_value = match constants.value {
                        Some(v) => v,
                        None => panic!("constants value is a None type when trying to update the height threshold which shouldn't happen!"),
                    };
                    if m[0] > m[1] {
                        if m[0] > normal_value {
                            final_constants.update_height_threshold(-1).unwrap()
                        }
                        else {
                            final_constants.update_height_threshold(0).unwrap()
                        }
                    }
                    else {
                        if m[1] > normal_value {
                            final_constants.update_height_threshold(1).unwrap()
                        }
                        else {
                            final_constants.update_height_threshold(0).unwrap()
                        }
                    }
                },
                Err(_) => final_constants.update_height_threshold(0).unwrap(),
            };
        }
        else {
            panic!("what the nani why is dy_list None for height_threshold?!");
        }

        if dy_list.len() > 10 {
            for i in 0..7 {
                if let Some(dy) = dy_list[i + 10] {
                    final_constants = final_constants.update_stored_piece_value(i, dy * OPT_RATE)
                        .unwrap(); //will never reach this state
                }
                else {
                    break;
                }
            }
        }

        let eval = run_test_trials(&final_constants);
        final_constants.value = Some(eval);
        OptimizationConstants::save_to_file(&final_constants).unwrap();
        constants = final_constants;
        count += 1;
        if count > 10 {
            break;
        }
    }

    // Run Trials w/ Gradient Ascent

    //fix the logic for height threshold

}

fn ga_constant<'a> (constants: &'a OptimizationConstants, constant_index: i8, s: &rayon::Scope) -> Result<Arc<Mutex<[f64;2]>>, &'a str> {
    let outputs = Arc::new(Mutex::new([0.0, 0.0]));
    let output_clone_pos = Arc::clone(&outputs);
    let output_clone_neg = Arc::clone(&outputs);
    let constants_pos;
    let constants_neg;
    match constant_index { 
        0 => {
            constants_pos = constants.update_hole_cost(TEST_STEP);
            constants_neg = constants.update_hole_cost(-TEST_STEP);
        }
        1 => {
            constants_pos = constants.update_hole_height_cost(TEST_STEP);
            constants_neg = constants.update_hole_height_cost(-TEST_STEP);
        }
        v @ 2..=5 => {
            constants_pos = constants.update_line_values(v as usize - 2, TEST_STEP);
            constants_neg = constants.update_line_values(v as usize - 2, -TEST_STEP);
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
            constants_pos = match constants.update_height_threshold(1) {
                Err(_) => constants.update_height_threshold(0).unwrap(),
                Ok(v) => v,
            };
            constants_neg = match constants.update_height_threshold(-1) {
                Err(_) => constants.update_height_threshold(0).unwrap(),
                Ok(v) => v,
            };
        }
        v @ 10..=16 => {
            constants_pos = constants.update_stored_piece_value(v as usize - 10, TEST_STEP)?;
            constants_neg = constants.update_stored_piece_value(v as usize - 10, -TEST_STEP)?;
        }
        _ => {
            panic!("Smh Bruh not a correct index");
        }
    }

    s.spawn(move |_| {
        let tmp = run_test_trials(&constants_pos);
        output_clone_pos.lock().unwrap()[0] = tmp;
    });

    s.spawn(move |_| {
        let tmp = run_test_trials(&constants_neg);
        output_clone_neg.lock().unwrap()[1] = tmp;
    });

    Ok(outputs)
}

#[cfg(test)]
mod tests {
    use rand::Rng;
    use super::OptimizationConstants;

    #[test]
    #[ignore]
    fn test_grab() {
        let file = "./optimization/test_results.txt";
        println!("{:#?}", OptimizationConstants::grab_specific_constants(file, Some(1)));
        println!("{:#?}", OptimizationConstants::grab_all_constants(file));
    }

    #[test]
    #[ignore]
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
        let stored_piece_value = Some([rng.gen_range(-range, range), rng.gen_range(-range, range), rng.gen_range(-range, range), rng.gen_range(-range, range), rng.gen_range(-range, range), rng.gen_range(-range, range), rng.gen_range(-range, range)]);

        let constants = OptimizationConstants {
            hole_cost,
            hole_height_cost,
            jagged_cost,
            height_cost,
            combo_value,
            line_values,
            height_threshold,
            stored_piece_value,

            value: None,
        };
        
        OptimizationConstants::save_to_file(&constants);
    }

    #[test]
    //#[ignore]
    fn optimize_test() {
        super::optimization(100.0);
    }
}

fn run_test_trials(constants: &OptimizationConstants) -> f64 {
    let mut average_attack = Arc::new(Mutex::new(0.0));
    rayon::scope(|s| {
        for i in 0..NUMBER_OF_TRIALS {
            let tmp_constants = constants.clone();
            let avg_attack_clone = Arc::clone(&average_attack);
            s.spawn(move |_| {
                match run_test(NUMBER_OF_MOVES, tmp_constants) {
                    TestResult::Lost(num_moves, attack) => {
                        let adjusted_attack = attack as f64 *  num_moves as f64 / NUMBER_OF_MOVES as f64;
                        println!("Meme Lost Smh: {}", num_moves);
                        println!("Attack");
                        println!("Adjusted attack: {}", adjusted_attack);
                        let mut att = avg_attack_clone.lock().unwrap();
                        *att += adjusted_attack as f64;
                    },
                    TestResult::Complete(attack) => {
                        println!("Finished with attack: {}", attack);
                        let mut att = avg_attack_clone.lock().unwrap();
                        *att += attack as f64;
                    },
                };
            });
        }
    });
    

    let average_attack = *average_attack.lock().unwrap()/(NUMBER_OF_TRIALS as f64);
    let average_attack_per_move = average_attack/(NUMBER_OF_MOVES as f64);
    average_attack_per_move
}

fn run_test(num_moves: usize, constants: OptimizationConstants) -> TestResult {
    let mut generator = PieceGenerator::new();
    let mut field = Field::new([generator.get_next_piece(), generator.get_next_piece(), generator.get_next_piece(), generator.get_next_piece(), generator.get_next_piece()], constants);
    
    for i in 0..num_moves {
        let suggested_move = match field.calculate_all_resulting_fields_scope(DEPTH, true) {
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
#[derive(Debug)]
struct OptimizationConstants {
    hole_cost: f64, 
    hole_height_cost: f64, 
    line_values: [f64;4], 
    jagged_cost: f64, 
    height_cost: f64, 
    combo_value: f64, 
    height_threshold: u8, 
    stored_piece_value: Option<[f64;7]>,

    value: Option<f64>
}

impl OptimizationConstants {
    fn new(hole_cost: f64, hole_height_cost: f64, line_values: [f64; 4], jagged_cost: f64, height_cost: f64, combo_value: f64, height_threshold: u8, stored_piece_value: Option<[f64; 7]>, value: Option<f64>) -> OptimizationConstants {
        OptimizationConstants {
            hole_cost,
            hole_height_cost,
            line_values,
            jagged_cost,
            height_cost, 
            combo_value, 
            height_threshold,
            stored_piece_value,
            
            value,
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
    fn update_height_threshold(&self, change: i8) -> Result<OptimizationConstants, &str> {
        let mut changed_constants = self.clone();
        if (changed_constants.height_threshold as i8 + change) < 0 {
            return Err("smh new height is not U8");
        }
        let new_value = changed_constants.height_threshold as i8 + change;
        changed_constants.height_threshold = new_value as u8;
        Ok(changed_constants)
    }
    fn update_stored_piece_value(&self, index: usize, change: f64) -> Result<OptimizationConstants, &str> {
        if self.stored_piece_value == None {
            return Err("no stored pieces values");
        }
        let mut changed_constants = self.clone();
        match changed_constants.stored_piece_value {
            None => return Err("impossible state"),
            Some(mut s) => {
                s[index] += change;
                changed_constants.stored_piece_value = Some(s);
            },
        }
        Ok(changed_constants)
    }
    fn save_to_file (constants: &OptimizationConstants) -> Result<(), std::io::Error> {
        let mut file;
        let file_name = format!("moves-{}-depth-{}-numtrials-{}-optrate-{}-teststep-{}-storedp-{}.csv", NUMBER_OF_MOVES, DEPTH, NUMBER_OF_TRIALS, OPT_RATE, TEST_STEP, STORED_PIECES);
        let joined_path = format!("./optimization/{}", file_name);
        if !std::path::Path::new(&joined_path).exists() {
            if !std::path::Path::new("./optimization").exists() {
                fs::create_dir("./optimization")?;
            }
            file = File::create(&joined_path)?;
            //file = OpenOptions::new().write(true).open(joined_path);
            let top_row = format!("Hole-Cost,Hole-Height-Cost,1-Line-Value,2-Line-Value,3-Line-Value,4-Line-Value,Jagged-Cost,Height-Cost,Combo-Value,Height-Threshold{},Result",if STORED_PIECES {
                let mut output = String::from("");
                for i in 0..7 {
                    output = format!("{},{}", output, format!("Stored-Piece-Value-{}", i));
                }
                output
            } else {
                String::from("")
            });
            file.write_all(top_row.as_bytes())?;
        }
        file = OpenOptions::new().append(true).open(&joined_path)?;
        let mut argument = format!("\n{},{}", constants.hole_cost, constants.hole_height_cost);
        for i in 0..4 {
            argument = format!("{},{}", argument, constants.line_values[i]);
        }
        argument = format!("{},{},{},{},{}", argument, constants.jagged_cost, constants.height_cost, constants.combo_value, constants.height_threshold);
        if let Some(v) = constants.stored_piece_value {
            for i in 0..7 {
                argument = format!("{},{}", argument, v[i]);
            }
        };
        argument = format!{"{},{}", argument, match constants.value {
            None => String::from(""),
            Some(r) => {
                let r = r.to_string();
                r
            }
        }};
        file.write_all(argument.as_bytes())?;
        Ok(())
    }
    fn grab_specific_constants (file_path: &str, index: Option<usize>) -> Result<OptimizationConstants, std::io::Error> {
        let chosen_line;
        let mut line_values = [0.0;4];
        let stored_pieces;
        let value;
        let file_string = fs::read_to_string(file_path).unwrap();
        let lines: Vec<&str> = file_string.split('\n').collect();
        let mut arr: Vec<&str> = if let Some(i) = index {
            if i >= lines.len() { 
                return Err(std::io::Error::new(ErrorKind::Other, "meme"));
            }
            chosen_line = lines[i];
            chosen_line.split(':').collect()
        }
        else {
            chosen_line = lines.last().unwrap();
            chosen_line.split(':').collect()
        };
        if arr.last() == Some(&"") {
            arr.pop();
        }
        println!("{:?}", arr);
        for i in 2..=5 {
            line_values[i-2] = arr[i].to_string().parse::<f64>().unwrap();
        }
        match arr.len() {
            10 => {
                stored_pieces = None;
                value = None;
            },
            11 => {
                stored_pieces = None;
                value = Some(arr[10].to_string().parse::<f64>().unwrap());
            },
            17 => {
                let mut stored_pieces_array = [0.0;7];
                for i in 10..17 { 
                    stored_pieces_array[i-10] = arr[i].to_string().parse::<f64>().unwrap();
                }
                stored_pieces = Some(stored_pieces_array);
                value = None;
            },
            18 => {
                println!("bruh");
                let mut stored_pieces_array = [0.0;7];
                for i in 10..17 {
                    stored_pieces_array[i-10] = arr[i].to_string().parse::<f64>().unwrap();
                }
                stored_pieces = Some(stored_pieces_array);
                value = Some(arr[17].to_string().parse::<f64>().unwrap());
            },
            _ => {
                panic!("Smh line mismatch");
            }
        }
        let constants = OptimizationConstants::new(arr[0].to_string().parse::<f64>().unwrap(), arr[1].to_string().parse::<f64>().unwrap(), line_values, arr[6].to_string().parse::<f64>().unwrap(), arr[7].to_string().parse::<f64>().unwrap(), arr[8].to_string().parse::<f64>().unwrap(), arr[9].to_string().parse::<u8>().unwrap(), stored_pieces, value); 
        Ok(constants)
    }
    fn grab_all_constants (file_path: &str) -> Result<Vec<OptimizationConstants>, std::io::Error> {
        let mut vec_const = vec![];
        let mut count = 1;
        loop {
            vec_const.push(match OptimizationConstants::grab_specific_constants(file_path, Some(count)) {
                Err(_) => break,
                Ok(v) => v,
            });
            count += 1;
        }
        Ok(vec_const)
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
    fn new(upcoming_pieces: [u8;5], constants: OptimizationConstants) -> Field {
        Field {
            field_state: Array::<u8, Ix2>::zeros((20, 10)),
            // Create a three-dimensional f64 array, initialized with zeros
            upcoming_pieces,
            stored_piece: None,
            value: 0.0,
            combo: 0,
            back_to_back: false,

            total_attack: 0,
            
            constants,
        }
    }
    fn from<'a>(prior_field: &Field, new_move: &Move, new_piece: &Piece) -> Result<Field, &'a str> {
        let tmp2;
        let mut new_piece = new_piece;
        let mut stored_piece_value = 0.0;
        let stored_piece = if new_move.store == true {
            match prior_field.stored_piece {
                Some(p) => {
                    let tmp = new_piece.id;
                    tmp2 = Piece::get_piece_from_id(p).unwrap().clone();
                    stored_piece_value = match prior_field.constants.stored_piece_value {
                        Some(arr) => arr[p as usize] - arr[new_piece.id as usize],
                        None => 0.0,
                    };
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
        f.value += stored_piece_value;
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
        let score = -(self.constants.jagged_cost * jaggedness as f64 + self.constants.hole_height_cost * hole_value as f64 + self.constants.hole_cost * hole_count as f64 + self.constants.height_cost * height_costs as f64) + self.value;
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
                let mut vec: Vec<Move> = piece.get_all_moves(false);
                vec.append(&mut Piece::get_piece_from_id(stored_piece).unwrap().get_all_moves(true));
                vec
            }
            else {
                let mut vec: Vec<Move> = vec![];
                vec.extend(piece.get_all_moves(false));
                vec
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
            Some(moves[move_id].clone())
        }
    }
    fn calculate_all_resulting_fields_TP(&self, depth: usize, stored: bool) -> Option<Move> {
        if depth == 0 {
            println!("{}",self.eval());
            None
        }
        else {
            let child_result = Arc::new(Mutex::new((0, f64::MIN)));
            let piece = Piece::get_piece_from_id(self.upcoming_pieces[0]).unwrap();
            let moves = if stored == true {if let Some(stored_piece) = self.stored_piece {
                let mut vec: Vec<Move> = piece.get_all_moves(false);
                vec.append(&mut Piece::get_piece_from_id(stored_piece).unwrap().get_all_moves(true));
                vec
            } else {
                piece.get_all_moves(false)
            }} else {
                piece.get_all_moves(false)
            };
            let original_depth = depth;
            let start = time::Instant::now();
            
            TP.install(|| {
                rayon::scope(|s| {
                    for i in 0..moves.len() {
                        let field = match Field::from(self, &moves[i], &piece) {
                            Ok(f) => f,
                            Err(_e) => continue,
                        };
                        let result_clone = Arc::clone(&child_result);
                        s.spawn(move |s| {
                            recursive_resulting_fields_TP(field, depth - 1, i, result_clone, original_depth);
                        })
                    }
                });
                
            });
            //println!("initial thread creation time: {:?}", start.elapsed());
            let start = time::Instant::now();  
            //println!("finished: {:?}", start.elapsed());
            let move_id = child_result.lock().unwrap().0;
            Some(moves[move_id])
        }
    }
    fn calculate_all_resulting_fields_scope(&self, depth: usize, stored: bool) -> Option<Move> {
        if depth == 0 {
            println!("{}",self.eval());
            None
        }
        else {
            let child_result = Arc::new(Mutex::new((0, f64::MIN)));
            let piece = Piece::get_piece_from_id(self.upcoming_pieces[0]).unwrap();
            let moves = if stored == true {if let Some(stored_piece) = self.stored_piece {
                let mut vec: Vec<Move> = piece.get_all_moves(false);
                vec.append(&mut Piece::get_piece_from_id(stored_piece).unwrap().get_all_moves(true));
                vec
            } else {
                piece.get_all_moves(false)
            }} else {
                piece.get_all_moves(false)
            };
            let original_depth = depth;
            
            rayon::scope(|s| {
                for i in 0..moves.len() {
                    let field = match Field::from(self, &moves[i], &piece) {
                        Ok(f) => f,
                        Err(_e) => continue,
                    };
                    let result_clone = Arc::clone(&child_result);
                    s.spawn(move |_| {
                        recursive_resulting_fields_TP(field, depth - 1, i, result_clone, original_depth);
                    })
                }
            });

            let move_id = child_result.lock().unwrap().0;
            Some(moves[move_id])
        }
    }
}

fn recursive_resulting_fields_TP(f: Field, depth: usize, move_id: usize, result: Arc<Mutex<(usize, f64)>>, original_depth: usize) {
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

        rayon::scope(|s| {
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
                s.spawn(move |s| {
                    recursive_resulting_fields_TP(field, depth - 1, move_id, result_clone, original_depth);
                })
            }
        });
        //println!("thread creation time: {:?}", start.elapsed());    
        let mut max_eval = result.lock().unwrap();
        let child_result = child_result.lock().unwrap().1;
        if max_eval.1 < child_result {
            *max_eval = (move_id, child_result);
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
    possible_moves_false: Vec<Move>,
    possible_moves_true: Vec<Move>,
}

impl Piece {
    fn get_all_moves(&self, store: bool) -> Vec<Move> {
        if store {
            self.possible_moves_true.clone()
        }
        else {
            self.possible_moves_false.clone()
        }
    }
    fn get_all_moves_prev(orientations: Vec<(i8, Array2::<u8>)>, store: bool) -> Vec<Move> {
        let mut moves = Vec::new();
        for i in 0..orientations.len() {
            for x in -orientations[i].0..(10 - orientations[i].1.shape()[1] as i8 - orientations[i].0 + 1) {
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
