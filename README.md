# AutoDiff: Higher-Order Automatic Differentiation in Idris

This library implements automatic differentiation in Idris with support for both forward and reverse mode. It features true automatic differentiation in both modes, capable of computing derivatives of any order for a wide range of mathematical functions.

## Overview

Automatic differentiation (AD) is a set of techniques to numerically evaluate the derivative of a function specified by a computer program. It's different from both symbolic differentiation and numerical differentiation (finite differences) by providing efficient, accurate derivatives without the overhead of symbolic manipulation or the numerical errors of finite differences.

This implementation features:
- **Forward mode** differentiation using dual numbers
- **Reverse mode** (backward) differentiation using computational graphs
- Support for higher-order derivatives
- Entirely implemented in pure Idris with no external dependencies
- Custom data structures to ensure compatibility across Idris versions

## Key Features

1. **Dual Approach**: Implements both forward-mode AD (efficient for functions with few inputs) and reverse-mode AD (efficient for functions with many outputs)

2. **Higher-Order Derivatives**: Support for computing derivatives of any order in reverse mode

3. **Pure Automatic Differentiation**: Uses dual number arithmetic in forward mode, avoiding explicit coding of differentiation rules

4. **Extensive Function Support**: Works with:
   - Polynomial functions
   - Trigonometric functions (sin, cos)
   - Exponential and logarithmic functions
   - Compositions and combinations of the above

5. **Type Safety**: Leverages Idris's strong type system to prevent runtime errors

## Quick Start

### Installation

Clone this repository and compile with Idris 2:

```bash
idris2 -o autodiff AutoDiff.idr
```

### Running the Examples

```bash
./build/exec/autodiff
```

### Using the Library

#### Forward Mode (using Dual Numbers)

```idris
import AutoDiff

-- Define a function using Dual type
myFunc : Dual -> Dual
myFunc x = square x + fromInteger 3 * x + fromInteger 2

-- Calculate derivative at x=2.0
derivative = forward_derivative myFunc 2.0  -- Get first derivative
```

#### Reverse Mode (using Computational Graphs)

```idris
import AutoDiff

-- Define a function using Expr type
myFunc : Expr -> Expr
myFunc x = AddExpr (AddExpr (MulExpr x x) (MulExpr (ConstExpr 3.0) x)) (ConstExpr 2.0)

-- Calculate derivatives at x=2.0
firstDeriv = backward_derivative myFunc 2.0 1   -- First derivative
secondDeriv = backward_derivative myFunc 2.0 2  -- Second derivative
thirdDeriv = backward_derivative myFunc 2.0 3   -- Third derivative
```

## Example Functions

The library includes 10 example functions demonstrating various use cases:

1. **Polynomial**: f(x) = x² + 3x + 2
2. **Trigonometric Product**: f(x) = sin(x) × cos(x)
3. **Compound Function**: f(x) = e^(x²) / (1 + x²)
4. **Higher-degree Polynomial**: f(x) = x³ - 5x² + 7x - 3
5. **Composition of Functions**: f(x) = sin(x²)
6. **Logarithmic Rational Function**: f(x) = log(x) / x
7. **Nested Trigonometric Functions**: f(x) = cos(sin(x))
8. **Exponential of Trigonometric Sum**: f(x) = exp(sin(x) + cos(x))
9. **Rational Function**: f(x) = x / (1 + x²)²
10. **Gaussian-related Function**: f(x) = x * exp(-x²/2)

## Technical Details

### Data Structures

- **Forward Mode**:
  - `Dual`: Represents a dual number (a + bε) where ε² = 0
  - The first component is the function value
  - The second component is the derivative

- **Reverse Mode**:
  - `Expr`: Expression tree for computational graph representation
  - Various node types for different operations (Add, Mul, Sin, etc.)

### Core Functions

- `forward_derivative`: Compute first derivative using dual numbers
- `backward_derivative`: Compute nth derivative using computational graphs
- `gradToExpr`: Generate new computational graphs for higher-order derivatives

### Implementation Notes

- Forward mode uses dual number arithmetic for true automatic differentiation
- Reverse mode generates computational graphs and applies the chain rule through these graphs
- Each mode has its strengths: forward mode is simpler for first derivatives, reverse mode handles higher-order derivatives

## Mathematical Background

### Forward Mode with Dual Numbers

Forward mode implements automatic differentiation using dual numbers, which are pairs (a, b) representing a + bε where ε² = 0. The algebra of dual numbers naturally gives rise to derivatives:

- Addition: (a, b) + (c, d) = (a+c, b+d)
- Multiplication: (a, b) * (c, d) = (a*c, a*d + b*c)
- Division: (a, b) / (c, d) = (a/c, (b*c - a*d)/(c*c))

When we initialize x = (x₀, 1) and compute f(x), the result is (f(x₀), f'(x₀)), giving us both the function value and its derivative.

### Reverse Mode with Computational Graphs

Reverse mode implements automatic differentiation by:
1. Constructing a computational graph representing the function evaluation
2. Computing gradients by backpropagating through this graph
3. For higher-order derivatives, building new computational graphs for each derivative level

Unlike traditional symbolic differentiation, this method doesn't generate simplified symbolic formulas, but rather creates executable computation graphs that numerically evaluate the derivatives.

## License

This project currently has no specified license. All rights are reserved by the author. If you intend to use, distribute, or modify this code, please contact the author for permission.

## Future Enhancements

- Support for higher-order derivatives in forward mode using tensor algebra
- Vector and matrix operations for multivariate differentiation
- Integration with numerical optimization algorithms
- Performance optimizations for large-scale computations
