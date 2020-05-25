

def gcd (a, b):
    if a==0: return b 
    if b==0: return a

    if a > b: return gcd(a-b, b) 
    else: return gcd(b-a, a) 



q = 71 
alpha = 33
Xb = 62 
Yb = (alpha**Xb) % q

i = 31 
Km = (Yb**i) % q
Ke = (alpha**i) % q 


M = 15 
C = Km*M % q 



Km_inverse = Ke**(q-1-Xb) % q


new_M = C*Km_inverse % q 

#________________________________________________________


p = 17
q = 11

n = p * q 
print("n:", n) 

temp = (p-1) * (q-1) 
print("alpha:", temp) 

e = 2 
for i in range(2, temp) :
    if gcd(temp, i) == 1:
        e = i 
        break
assert gcd(temp, e)==1, "value of e is incorrect: e=={:d}".format(e) 
print("e:", e) 

d = 1
for i in range(1, temp) :
    d = i
    if (d*e) % temp == 1 : break 
assert (d*e) % temp ==1, "value of d is incorrect: d=={:d}".format(d) 
print("d:", d) 

public_key = [e, n] 
private_key = [d, n] 


M = 15

encrypted = (M**public_key[0]) % public_key[1]
print("encrypted:", encrypted) 

decrypted = (encrypted**private_key[0]) % private_key[1]
print("decrypted:", decrypted) 