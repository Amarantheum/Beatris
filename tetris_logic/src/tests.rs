use rayon::prelude::*;
use ndarray::prelude::*;
use ndarray::stack;
use pyo3::exceptions;
use std::sync::Arc;
use std::fmt::Display;
use std::fmt;
use std::sync::Mutex;
use std::thread;
use std::sync::atomic::{AtomicI32, AtomicUsize, AtomicU8, Ordering};
use std::time;


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
    
    //try adding a timer here and if it takes forever to run tetris_logic functions, that means it's doing this every time
}

// Evaluation Constants
static mut BOX_UNIT: AtomicUsize = AtomicUsize::new(24);
static mut HOLE_COST: AtomicI32 = AtomicI32::new(10);
static mut HOLE_HEIGHT_COST: AtomicI32 = AtomicI32::new(2);
static mut LINE_VALUE: AtomicI32 = AtomicI32::new(25);
static mut JAGGED_COST: AtomicI32 = AtomicI32::new(2);
static mut HEIGHT_THRESHOLD: AtomicU8 = AtomicU8::new(8);
static mut HEIGHT_COST: AtomicI32 = AtomicI32::new(4);

// Storing Pieces Constants
static mut I_BLOCK: AtomicI32 = AtomicI32::new(2);
static mut O_BLOCK: AtomicI32 = AtomicI32::new(0);
static mut T_BLOCK: AtomicI32 = AtomicI32::new(1);
static mut S_BLOCK: AtomicI32 = AtomicI32::new(0);
static mut Z_BLOCK: AtomicI32 = AtomicI32::new(0);
static mut J_BLOCK: AtomicI32 = AtomicI32::new(0);
static mut L_BLOCK: AtomicI32 = AtomicI32::new(0);

//can't be greater than 9 for the moment
const MOVES_PER_THREAD: usize = 20;

fn get_box_unit() -> usize {
    unsafe {
        BOX_UNIT.load(Ordering::Relaxed)
    }
}
fn get_hole_cost() -> i32 {
    unsafe {
        HOLE_COST.load(Ordering::Relaxed)
    }
}
fn get_hole_height_cost() -> i32 {
    unsafe {
        HOLE_HEIGHT_COST.load(Ordering::Relaxed)
    }
}
fn get_line_value() -> i32 {
    unsafe {
        LINE_VALUE.load(Ordering::Relaxed)
    }
}
fn get_jagged_cost() -> i32 {
    unsafe {
        JAGGED_COST.load(Ordering::Relaxed)
    }
}
fn get_height_threshold() -> u8 {
    unsafe {
        HEIGHT_THRESHOLD.load(Ordering::Relaxed)
    }
}
fn get_height_cost() -> i32 {
    unsafe {
        HEIGHT_COST.load(Ordering::Relaxed)
    }
}

#[derive(Clone)]
struct Field {
    field_state: Array::<u8, Ix2>,
    upcoming_pieces: [u8;5],
    stored_piece: Option<u8>,
    value: f32,
    combo: u8,
}

impl Field {
    fn new(upcoming_pieces: [u8;5]) -> Field {
        Field {
            field_state: Array::<u8, Ix2>::zeros((20, 10)),
            // Create a three-dimensional f64 array, initialized with zeros
            upcoming_pieces,
            stored_piece: None,
            value: 0.0,
            combo: 0,
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
            upcoming_pieces: prior_field.upcoming_pieces.clone(), 
            

            value: prior_field.value.clone(),
            combo: prior_field.combo.clone(),
        };
        Field::clean_sent_lines(&mut f);
        Ok(f)
    }
    fn eval_sent_lines(attack:usize) -> f32 {
        match attack {
            0 => 0.0, 
            1 => -0.5, 
            2 | 3 => (attack - 1).pow(2) as f32,
            4 => 16.0,
            _ => (attack).pow(2) as f32, 
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
            field.combo += 1;
        }
        else {
            field.combo = 0;
        }
        field.value += get_line_value() as f32 * Field::eval_sent_lines(count) as f32;
    }
    fn eval(&self) -> f32 {
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
                    if (y_values[x] as u8) < (20 - get_height_threshold() as u8) {
                        height_costs += 20 - y_values[x] - get_height_threshold() as usize;
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
        let score = -(get_jagged_cost() * jaggedness as i32 + get_hole_height_cost() * hole_value as i32 + get_hole_cost() * hole_count as i32 + get_height_cost() * height_costs as i32) as f32;
        score
    }
    
    fn calculate_all_resulting_fields(&self, depth: usize) -> Option<Move> {
        if depth == 0 {
            println!("{}",self.eval());
            None
        }
        else {
            let child_result = Arc::new(Mutex::new((0, f32::MIN)));
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
            let child_result = Arc::new(Mutex::new((0, f32::MIN)));
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
                        let child_result_thread = Arc::new(Mutex::new((0, f32::MIN)));
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
fn recursive_resulting_fields(f: Field, depth: usize, move_id: usize, result: Arc<Mutex<(usize, f32)>>, original_depth: usize) {
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
        let child_result = Arc::new(Mutex::new((0, f32::MIN)));
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

fn recursive_resulting_fields_new(f: Field, depth: usize, move_id: usize, result: Arc<Mutex<(usize, f32)>>, original_depth: usize) {
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
        let child_result = Arc::new(Mutex::new((0, f32::MIN)));
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
                    let child_result_thread = Arc::new(Mutex::new((0, f32::MIN)));
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
        write!(f, "(Rotation: {}, Position: {}, Store: {})", self.rotation, self.position, self.store)
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


#[cfg(test)]
mod tests {
    use ndarray::prelude::*;

    use std::sync::Arc;
    use std::thread;
    use super::Field;
    use super::Move;
    use super::Piece;
    use std::time;
    use std::thread::sleep;
    use rand::Rng;



    #[test]
    fn benchmark() {
        let piece = Piece::get_piece_from_id(5).unwrap();
        let start = time::Instant::now();
        piece.get_all_moves(true);
        println!("Elapsed: {:?}", start.elapsed());
    }

    #[test]
    //#[ignore]
    fn from_test() {
        let mut total1 = 0;
        let mut total2 = 0;
        for i in 0..100 {
            
            let mut field = Field::new([rand::thread_rng().gen_range(0,7),rand::thread_rng().gen_range(0,7),rand::thread_rng().gen_range(0,7),5,4]);
            //let mut field = Field::new([5,2,5,6,7]);
            let m = Move::from((0,0,true));
            field.stored_piece = Some(rand::thread_rng().gen_range(0,7));
            let start = time::Instant::now();
            let m1 = match field.calculate_all_resulting_fields(2){
                Some(m) => m,
                None => panic!("bruh"),
            };
            total1 += start.elapsed().as_millis();
            //sleep(time::Duration::from_secs(3));
            let start = time::Instant::now();
            let m2 = match field.calculate_all_resulting_fields_new(2, true){
                Some(m) => m,
                None => panic!("bruh"),
            };
            total2 += start.elapsed().as_millis();
        }
        println!("first: {}, second: {}, diff: {}, ratio: {}", total1, total2, total1 as i128 - total2 as i128, total1 as f32/total2 as f32)
    }
}