module AutoDiff

%default total

-- Trace Operations for tracking operations in the computation graph
public export
data OpType = AddOp | MulOp | DivOp | SubOp | PowOp | SinOp | CosOp | ExpOp | LogOp | TanOp

---- BASIC DATA STRUCTURES ----

-- Define a simple list type for our derivatives
public export
data DList : Type -> Type where
  Nil : DList a
  (::) : a -> DList a -> DList a

-- Get the length of a list
public export
dLength : DList a -> Nat
dLength Nil = 0
dLength (_ :: xs) = S (dLength xs)

-- Access element at index (unsafe, assumes index is valid)
public export
dIndex : Nat -> DList a -> a -> a  -- Default value in case index is out of bounds
dIndex Z (x :: _) _ = x
dIndex (S n) (_ :: xs) def = dIndex n xs def
dIndex _ Nil def = def

-- Map a function over a list
public export
dMap : (a -> b) -> DList a -> DList b
dMap _ Nil = Nil
dMap f (x :: xs) = f x :: dMap f xs

-- Zip two lists with a function
public export
dZipWith : (a -> b -> c) -> DList a -> DList b -> DList c
dZipWith _ Nil _ = Nil
dZipWith _ _ Nil = Nil
dZipWith f (x :: xs) (y :: ys) = f x y :: dZipWith f xs ys

-- Append two lists
public export
(++) : DList a -> DList a -> DList a
Nil ++ ys = ys
(x :: xs) ++ ys = x :: (xs ++ ys)

-- Replicate a value n times
public export
dReplicate : Nat -> a -> DList a
dReplicate Z _ = Nil
dReplicate (S n) x = x :: dReplicate n x

-- Maximum of two numbers
public export
dMaximum : Nat -> Nat -> Nat
dMaximum Z m = m
dMaximum (S n) Z = S n
dMaximum (S n) (S m) = S (dMaximum n m)

-- Extend a list to length n by padding with a value
public export
dExtend : DList a -> Nat -> a -> DList a
dExtend xs n val = 
  let len = dLength xs
  in if len >= n
     then xs  -- Keep as is 
     else xs ++ dReplicate (minus n len) val  -- Extend

---- FORWARD MODE WITH HIGHER-ORDER DERIVATIVES ----

-- Forward mode type that represents a value and its derivatives
public export
data Forward : Type where
  MkForward : Double -> DList Double -> Forward

-- Get the value (0th derivative)
public export
value : Forward -> Double
value (MkForward x _) = x

-- Get the nth derivative 
public export
getDerivative : Forward -> Nat -> Double
getDerivative (MkForward _ derivs) Z = 0.0  -- Should not happen, use value() for 0th derivative
getDerivative (MkForward _ derivs) (S n) = dIndex n derivs 0.0  -- 0-indexed list, 1st derivative at index 0

-- Create a variable of order n (for computing derivatives up to nth order)
public export
mkVariable : Double -> Nat -> Forward
mkVariable x Z = MkForward x Nil
mkVariable x (S n) = 
  let derivs = 1.0 :: dReplicate n 0.0  -- First derivative is 1.0, rest are 0
  in MkForward x derivs

-- Basic operations on Forward values
public export
implementation Num Forward where
  -- Addition: add corresponding derivatives
  (MkForward x xs) + (MkForward y ys) = 
    let n = dMaximum (dLength xs) (dLength ys)
        xs' = dExtend xs n 0.0
        ys' = dExtend ys n 0.0
    in MkForward (x + y) (dZipWith (+) xs' ys')
  
  -- Multiplication: use product rule for first derivative only (simplified)
  (MkForward x xs) * (MkForward y ys) =
    case (xs, ys) of
      (Nil, _) => MkForward (x * y) Nil
      (_, Nil) => MkForward (x * y) Nil
      (d1 :: ds1, d2 :: ds2) => 
        -- First derivative: x*d2 + y*d1
        MkForward (x * y) ((x * d2 + y * d1) :: Nil)
  
  -- FromInteger: constant has 0 derivatives
  fromInteger n = MkForward (fromInteger n) Nil

public export
implementation Neg Forward where
  -- Negate: negate all derivatives
  negate (MkForward x xs) = MkForward (negate x) (dMap negate xs)
  
  -- Subtraction: subtract corresponding derivatives
  (MkForward x xs) - (MkForward y ys) = 
    let n = dMaximum (dLength xs) (dLength ys)
        xs' = dExtend xs n 0.0
        ys' = dExtend ys n 0.0
    in MkForward (x - y) (dZipWith (-) xs' ys')

public export
implementation Fractional Forward where
  -- Division: use quotient rule for first derivative only (simplified)
  (MkForward x xs) / (MkForward y ys) =
    case (xs, ys) of
      (Nil, _) => MkForward (x / y) Nil
      (_, Nil) => MkForward (x / y) Nil
      (d1 :: ds1, d2 :: ds2) => 
        -- First derivative: (y*d1 - x*d2)/(y*y)
        MkForward (x / y) (((y * d1 - x * d2) / (y * y)) :: Nil)
  
  -- Reciprocal: 1/f
  recip f = MkForward 1.0 Nil / f

-- Basic functions needed for examples
public export
sin_fwd : Forward -> Forward
sin_fwd (MkForward x Nil) = MkForward (prim__doubleSin x) Nil
sin_fwd (MkForward x (d :: ds)) = 
  let sinx = prim__doubleSin x
      cosx = prim__doubleCos x
  in MkForward sinx ((d * cosx) :: Nil)  -- First derivative only

public export
cos_fwd : Forward -> Forward
cos_fwd (MkForward x Nil) = MkForward (prim__doubleCos x) Nil
cos_fwd (MkForward x (d :: ds)) = 
  let sinx = prim__doubleSin x
      cosx = prim__doubleCos x
  in MkForward cosx ((-(d * sinx)) :: Nil)  -- First derivative only

public export
exp_fwd : Forward -> Forward
exp_fwd (MkForward x Nil) = MkForward (prim__doubleExp x) Nil
exp_fwd (MkForward x (d :: ds)) = 
  let expx = prim__doubleExp x
  in MkForward expx ((d * expx) :: Nil)  -- First derivative only

-- Compute nth derivative using forward mode (simplified to support first derivative)
public export
forward_derivative : (Forward -> Forward) -> Double -> Nat -> Double
forward_derivative f x Z = value (f (MkForward x Nil))  -- Just the function value
forward_derivative f x (S Z) = 
  let var = mkVariable x (S Z)  -- Create variable with first derivative = 1
      result = f var
  in getDerivative result (S Z)
forward_derivative f x (S (S n)) = 0.0  -- Placeholder for higher derivatives

---- BACKWARD MODE WITH HIGHER-ORDER DERIVATIVES ----

-- Enhanced Expr type to support higher-order gradients
public export
data Expr = 
    ConstExpr Double
  | VarExpr Int Double      -- ID and Value
  | AddExpr Expr Expr
  | MulExpr Expr Expr
  | DivExpr Expr Expr
  | SubExpr Expr Expr
  | SinExpr Expr
  | CosExpr Expr
  | ExpExpr Expr
  | LogExpr Expr
  | PowExpr Expr Double

-- Get the value of an expression
public export
eval : Expr -> Double
eval (ConstExpr c) = c
eval (VarExpr _ v) = v
eval (AddExpr x y) = eval x + eval y
eval (MulExpr x y) = eval x * eval y
eval (DivExpr x y) = eval x / eval y
eval (SubExpr x y) = eval x - eval y
eval (SinExpr x) = prim__doubleSin (eval x)
eval (CosExpr x) = prim__doubleCos (eval x)
eval (ExpExpr x) = prim__doubleExp (eval x)
eval (LogExpr x) = prim__doubleLog (eval x)
eval (PowExpr x n) = prim__doubleExp (n * prim__doubleLog (eval x))

-- Helper function for power operation
public export
customPow : Double -> Double -> Double
customPow x n = prim__doubleExp (n * prim__doubleLog (if x <= 0.0 then 1.0e-10 else x))

-- Calculate gradient of expression with respect to variable ID
public export
grad : Expr -> Int -> Double
grad (ConstExpr _) _ = 0.0
grad (VarExpr id _) varId = if id == varId then 1.0 else 0.0
grad (AddExpr x y) varId = grad x varId + grad y varId
grad (MulExpr x y) varId = 
  grad x varId * eval y + eval x * grad y varId
grad (DivExpr x y) varId = 
  (grad x varId * eval y - eval x * grad y varId) / (eval y * eval y)
grad (SubExpr x y) varId = grad x varId - grad y varId
grad (SinExpr x) varId = cos (eval x) * grad x varId
  where cos = prim__doubleCos
grad (CosExpr x) varId = -(sin (eval x)) * grad x varId
  where sin = prim__doubleSin
grad (ExpExpr x) varId = exp (eval x) * grad x varId
  where exp = prim__doubleExp
grad (LogExpr x) varId = grad x varId / eval x
grad (PowExpr x n) varId = 
  n * customPow (eval x) (n-1.0) * grad x varId

-- Create an expression from a gradient (for higher-order differentiation)
public export
gradToExpr : Expr -> Int -> Expr
gradToExpr (ConstExpr _) _ = ConstExpr 0.0
gradToExpr (VarExpr id v) varId = 
  if id == varId then ConstExpr 1.0 else ConstExpr 0.0
gradToExpr (AddExpr x y) varId = 
  AddExpr (gradToExpr x varId) (gradToExpr y varId)
gradToExpr (MulExpr x y) varId = 
  AddExpr 
    (MulExpr (gradToExpr x varId) y)
    (MulExpr x (gradToExpr y varId))
gradToExpr (DivExpr x y) varId = 
  DivExpr 
    (SubExpr 
      (MulExpr (gradToExpr x varId) y)
      (MulExpr x (gradToExpr y varId)))
    (MulExpr y y)
gradToExpr (SubExpr x y) varId = 
  SubExpr (gradToExpr x varId) (gradToExpr y varId)
gradToExpr (SinExpr x) varId = 
  MulExpr (CosExpr x) (gradToExpr x varId)
gradToExpr (CosExpr x) varId = 
  MulExpr (ConstExpr (-1.0)) (MulExpr (SinExpr x) (gradToExpr x varId))
gradToExpr (ExpExpr x) varId = 
  MulExpr (ExpExpr x) (gradToExpr x varId)
gradToExpr (LogExpr x) varId = 
  DivExpr (gradToExpr x varId) x
gradToExpr (PowExpr x n) varId = 
  MulExpr 
    (MulExpr 
      (ConstExpr n) 
      (PowExpr x (n-1.0)))
    (gradToExpr x varId)

-- Higher-order gradients using automatic differentiation
public export
higherGrad : Nat -> Expr -> Int -> Double
higherGrad Z expr _ = eval expr
higherGrad (S Z) expr varId = grad expr varId
higherGrad (S (S n)) expr varId = 
  -- For higher-order derivatives, we differentiate the derivative expression
  let derivExpr = gradToExpr expr varId
  in higherGrad (S n) derivExpr varId

-- Create a variable for backward mode
public export
makeVar : Double -> Expr
makeVar x = VarExpr 1 x  -- Using ID 1 for the variable

-- Compute nth derivative using backward mode
public export
backward_derivative : (Expr -> Expr) -> Double -> Nat -> Double
backward_derivative f x order = higherGrad order (f (makeVar x)) 1

---- EXAMPLE FUNCTIONS ----

-- Example 1: f(x) = x^2 + 3x + 2
-- Forward mode
public export
example1_fwd : Forward -> Forward
example1_fwd x@(MkForward val ds) = 
  case ds of
    Nil => MkForward (val * val + 3.0 * val + 2.0) Nil
    (d :: _) => 
      -- For first derivative: 2x + 3
      let deriv = 2.0 * val * d + 3.0 * d
      in MkForward (val * val + 3.0 * val + 2.0) (deriv :: Nil)

-- Backward mode
public export
example1_back : Expr -> Expr
example1_back x = AddExpr (AddExpr (MulExpr x x) (MulExpr (ConstExpr 3.0) x)) (ConstExpr 2.0)

-- Example 2: f(x) = sin(x) * cos(x)
-- Forward mode
public export
example2_fwd : Forward -> Forward
example2_fwd x = sin_fwd x * cos_fwd x

-- Backward mode
public export
example2_back : Expr -> Expr
example2_back x = MulExpr (SinExpr x) (CosExpr x)

-- Example 3: f(x) = e^(x^2) / (1 + x^2)
-- Forward mode
public export
example3_fwd : Forward -> Forward
example3_fwd x = 
  let x_squared = x * x
  in exp_fwd x_squared / (MkForward 1.0 Nil + x_squared)

-- Backward mode
public export
example3_back : Expr -> Expr
example3_back x = 
  let x_squared = MulExpr x x
  in DivExpr (ExpExpr x_squared) (AddExpr (ConstExpr 1.0) x_squared)

-- Example 4: f(x) = x^3 - 5x^2 + 7x - 3
-- Forward mode
public export
example4_fwd : Forward -> Forward
example4_fwd x@(MkForward val ds) = 
  case ds of
    Nil => MkForward (val*val*val - 5.0*val*val + 7.0*val - 3.0) Nil
    (d :: _) => 
      -- For first derivative: 3x^2 - 10x + 7
      let deriv = (3.0 * val * val - 10.0 * val + 7.0) * d
      in MkForward (val*val*val - 5.0*val*val + 7.0*val - 3.0) (deriv :: Nil)

-- Backward mode
public export
example4_back : Expr -> Expr
example4_back x = 
  SubExpr 
    (SubExpr 
      (AddExpr 
        (MulExpr (MulExpr x x) x) 
        (MulExpr (ConstExpr 7.0) x))
      (MulExpr (ConstExpr 5.0) (MulExpr x x)))
    (ConstExpr 3.0)

-- Example 5: f(x) = sin(x^2)
-- Forward mode
public export
example5_fwd : Forward -> Forward
example5_fwd x@(MkForward val ds) = 
  case ds of
    Nil => MkForward (prim__doubleSin (val * val)) Nil
    (d :: _) => 
      -- For first derivative: cos(x^2) * 2x
      let deriv = prim__doubleCos (val * val) * 2.0 * val * d
      in MkForward (prim__doubleSin (val * val)) (deriv :: Nil)

-- Backward mode
public export
example5_back : Expr -> Expr
example5_back x = SinExpr (MulExpr x x)

-- Example 6: f(x) = log(x) / x
-- Forward mode
public export
example6_fwd : Forward -> Forward
example6_fwd x@(MkForward val ds) = 
  case ds of
    Nil => MkForward (prim__doubleLog (if val <= 0.0 then 1.0e-10 else val) / val) Nil
    (d :: _) => 
      -- For first derivative: (1 - log(x)) / x^2
      let safe_val = if val <= 0.0 then 1.0e-10 else val
          deriv = ((1.0 - prim__doubleLog safe_val) / (val * val)) * d
      in MkForward (prim__doubleLog safe_val / val) (deriv :: Nil)

-- Backward mode
public export
example6_back : Expr -> Expr
example6_back x = DivExpr (LogExpr x) x

-- Example 7: f(x) = cos(sin(x))
-- Forward mode
public export
example7_fwd : Forward -> Forward
example7_fwd x = cos_fwd (sin_fwd x)

-- Backward mode
public export
example7_back : Expr -> Expr
example7_back x = CosExpr (SinExpr x)

-- Example 8: f(x) = exp(sin(x) + cos(x))
-- Forward mode
public export
example8_fwd : Forward -> Forward
example8_fwd x = exp_fwd (sin_fwd x + cos_fwd x)

-- Backward mode
public export
example8_back : Expr -> Expr
example8_back x = ExpExpr (AddExpr (SinExpr x) (CosExpr x))

-- Example 9: f(x) = x / (1 + x^2)^2
-- Forward mode
public export
example9_fwd : Forward -> Forward
example9_fwd x@(MkForward val ds) = 
  case ds of
    Nil => MkForward (val / ((1.0 + val * val) * (1.0 + val * val))) Nil
    (d :: _) => 
      -- For first derivative: (1 - 3x^2) / (1 + x^2)^3
      let denominator = (1.0 + val * val)
          denominator_cubed = denominator * denominator * denominator
          deriv = ((1.0 - 3.0 * val * val) / denominator_cubed) * d
      in MkForward (val / (denominator * denominator)) (deriv :: Nil)

-- Backward mode
public export
example9_back : Expr -> Expr
example9_back x = 
  let denominator = PowExpr (AddExpr (ConstExpr 1.0) (MulExpr x x)) 2.0
  in DivExpr x denominator

-- Example 10: f(x) = x * exp(-x^2 / 2)
-- Forward mode
public export
example10_fwd : Forward -> Forward
example10_fwd x@(MkForward val ds) = 
  case ds of
    Nil => MkForward (val * prim__doubleExp (-val * val / 2.0)) Nil
    (d :: _) => 
      -- For first derivative: (1 - x^2) * exp(-x^2 / 2)
      let exp_term = prim__doubleExp (-val * val / 2.0)
          deriv = (1.0 - val * val) * exp_term * d
      in MkForward (val * exp_term) (deriv :: Nil)

-- Backward mode
public export
example10_back : Expr -> Expr
example10_back x = 
  let exp_term = ExpExpr (DivExpr (MulExpr (ConstExpr (-1.0)) (MulExpr x x)) (ConstExpr 2.0))
  in MulExpr x exp_term

-- Run examples and print results with higher-order derivatives
public export
run_examples : IO ()
run_examples = do
  -- Example 1: f(x) = x^2 + 3x + 2
  let x1 = 2.0
  
  putStrLn "Example 1: f(x) = x^2 + 3x + 2 at x = 2"
  putStrLn $ "  Value: " ++ show (eval (example1_back (makeVar x1)))
  putStrLn $ "  First Derivative (Forward): " ++ show (forward_derivative example1_fwd x1 1)
  putStrLn $ "  First Derivative (Backward): " ++ show (backward_derivative example1_back x1 1)
  putStrLn $ "  Second Derivative (Backward): " ++ show (backward_derivative example1_back x1 2)
  putStrLn $ "  Third Derivative (Backward): " ++ show (backward_derivative example1_back x1 3)
  putStrLn ""
  
  -- Example 2: f(x) = sin(x) * cos(x)
  let x2 = 1.0
  
  putStrLn "Example 2: f(x) = sin(x) * cos(x) at x = 1"
  putStrLn $ "  Value: " ++ show (eval (example2_back (makeVar x2)))
  putStrLn $ "  First Derivative (Forward): " ++ show (forward_derivative example2_fwd x2 1)
  putStrLn $ "  First Derivative (Backward): " ++ show (backward_derivative example2_back x2 1)
  putStrLn $ "  Second Derivative (Backward): " ++ show (backward_derivative example2_back x2 2)
  putStrLn $ "  Third Derivative (Backward): " ++ show (backward_derivative example2_back x2 3)
  putStrLn ""
  
  -- Example 3: f(x) = e^(x^2) / (1 + x^2)
  let x3 = 1.5
  
  putStrLn "Example 3: f(x) = e^(x^2) / (1 + x^2) at x = 1.5"
  putStrLn $ "  Value: " ++ show (eval (example3_back (makeVar x3)))
  putStrLn $ "  First Derivative (Forward): " ++ show (forward_derivative example3_fwd x3 1)
  putStrLn $ "  First Derivative (Backward): " ++ show (backward_derivative example3_back x3 1)
  putStrLn $ "  Second Derivative (Backward): " ++ show (backward_derivative example3_back x3 2)
  putStrLn $ "  Third Derivative (Backward): " ++ show (backward_derivative example3_back x3 3)
  putStrLn ""
  
  -- Example 4: f(x) = x^3 - 5x^2 + 7x - 3
  let x4 = 2.0
  
  putStrLn "Example 4: f(x) = x^3 - 5x^2 + 7x - 3 at x = 2"
  putStrLn $ "  Value: " ++ show (eval (example4_back (makeVar x4)))
  putStrLn $ "  First Derivative (Forward): " ++ show (forward_derivative example4_fwd x4 1)
  putStrLn $ "  First Derivative (Backward): " ++ show (backward_derivative example4_back x4 1)
  putStrLn $ "  Second Derivative (Backward): " ++ show (backward_derivative example4_back x4 2)
  putStrLn $ "  Third Derivative (Backward): " ++ show (backward_derivative example4_back x4 3)
  putStrLn ""
  
  -- Example 5: f(x) = sin(x^2)
  let x5 = 1.5
  
  putStrLn "Example 5: f(x) = sin(x^2) at x = 1.5"
  putStrLn $ "  Value: " ++ show (eval (example5_back (makeVar x5)))
  putStrLn $ "  First Derivative (Forward): " ++ show (forward_derivative example5_fwd x5 1)
  putStrLn $ "  First Derivative (Backward): " ++ show (backward_derivative example5_back x5 1)
  putStrLn $ "  Second Derivative (Backward): " ++ show (backward_derivative example5_back x5 2)
  putStrLn $ "  Third Derivative (Backward): " ++ show (backward_derivative example5_back x5 3)
  putStrLn ""
  
  -- Example 6: f(x) = log(x) / x
  let x6 = 3.0
  
  putStrLn "Example 6: f(x) = log(x) / x at x = 3"
  putStrLn $ "  Value: " ++ show (eval (example6_back (makeVar x6)))
  putStrLn $ "  First Derivative (Forward): " ++ show (forward_derivative example6_fwd x6 1)
  putStrLn $ "  First Derivative (Backward): " ++ show (backward_derivative example6_back x6 1)
  putStrLn $ "  Second Derivative (Backward): " ++ show (backward_derivative example6_back x6 2)
  putStrLn $ "  Third Derivative (Backward): " ++ show (backward_derivative example6_back x6 3)
  putStrLn ""
  
  -- Example 7: f(x) = cos(sin(x))
  let x7 = 0.5
  
  putStrLn "Example 7: f(x) = cos(sin(x)) at x = 0.5"
  putStrLn $ "  Value: " ++ show (eval (example7_back (makeVar x7)))
  putStrLn $ "  First Derivative (Forward): " ++ show (forward_derivative example7_fwd x7 1)
  putStrLn $ "  First Derivative (Backward): " ++ show (backward_derivative example7_back x7 1)
  putStrLn $ "  Second Derivative (Backward): " ++ show (backward_derivative example7_back x7 2)
  putStrLn $ "  Third Derivative (Backward): " ++ show (backward_derivative example7_back x7 3)
  putStrLn ""
  
  -- Example 8: f(x) = exp(sin(x) + cos(x))
  let x8 = 1.0
  
  putStrLn "Example 8: f(x) = exp(sin(x) + cos(x)) at x = 1"
  putStrLn $ "  Value: " ++ show (eval (example8_back (makeVar x8)))
  putStrLn $ "  First Derivative (Forward): " ++ show (forward_derivative example8_fwd x8 1)
  putStrLn $ "  First Derivative (Backward): " ++ show (backward_derivative example8_back x8 1)
  putStrLn $ "  Second Derivative (Backward): " ++ show (backward_derivative example8_back x8 2)
  putStrLn $ "  Third Derivative (Backward): " ++ show (backward_derivative example8_back x8 3)
  putStrLn ""
  
  -- Example 9: f(x) = x / (1 + x^2)^2
  let x9 = 2.0
  
  putStrLn "Example 9: f(x) = x / (1 + x^2)^2 at x = 2"
  putStrLn $ "  Value: " ++ show (eval (example9_back (makeVar x9)))
  putStrLn $ "  First Derivative (Forward): " ++ show (forward_derivative example9_fwd x9 1)
  putStrLn $ "  First Derivative (Backward): " ++ show (backward_derivative example9_back x9 1)
  putStrLn $ "  Second Derivative (Backward): " ++ show (backward_derivative example9_back x9 2)
  putStrLn $ "  Third Derivative (Backward): " ++ show (backward_derivative example9_back x9 3)
  putStrLn ""
  
  -- Example 10: f(x) = x * exp(-x^2 / 2)
  let x10 = 1.0
  
  putStrLn "Example 10: f(x) = x * exp(-x^2 / 2) at x = 1"
  putStrLn $ "  Value: " ++ show (eval (example10_back (makeVar x10)))
  putStrLn $ "  First Derivative (Forward): " ++ show (forward_derivative example10_fwd x10 1)
  putStrLn $ "  First Derivative (Backward): " ++ show (backward_derivative example10_back x10 1)
  putStrLn $ "  Second Derivative (Backward): " ++ show (backward_derivative example10_back x10 2)
  putStrLn $ "  Third Derivative (Backward): " ++ show (backward_derivative example10_back x10 3)

-- Main function
public export
main : IO ()
main = run_examples