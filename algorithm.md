# ToFUL
The **Tool for Uncertainity** used to calculate moments

# Algorithm: Moment & Probability Calculator  

---

## 1. Ask the user for the nature of the variable  
-  “Discrete (DRV) or Continuous (CRV)?” and store the choice.  

---

## 2. Get the Range  
- **DRV:** Accept a finite or infinite list such as `{1, 2, 3, …}`.  
- **CRV:** Accept an interval such as `(a,b)`, `[a,∞)`, etc.  

---

## 3. Accept the Probability Function  
- **Example:**  
  - DRV: `0.5 if x==0 else 0.3 if x==1 else 0.2`  
  - CRV: `2*x if 0<=x<=1 else 0`  
-  Allow functions such as `factorial`, `sqrt`, `exp`, `log`, `sin`, etc.  

---

## 4. Ask for the Moment About  
Offer three choices:  
1. About any value `a` (then prompt for `a`)  
2. About the origin (`a = 0`)  
3. About the mean (`a = μ`)  

---

## 5. Compute the Reference Value `a`  
- If **about the mean**:  

  - **DRV:**  
    $$\mu = \sum_{x} x \cdot f(x)$$  

  - **CRV:**  
    $$\mu = \int_{-\infty}^{\infty} x \cdot f(x)\, dx$$  

---

## 6. Compute the r-th Moment  

- **DRV (Discrete):**  
  $$\mu_r(a) = \mathbb{E}[(X-a)^r] = \sum_{x} (x-a)^r \, P(X=x)$$  

- **CRV (Continuous):**  
  $$\mu_r(a) = \mathbb{E}[(X-a)^r] = \int_{-\infty}^{\infty} (x-a)^r \, f(x)\, dx$$  


