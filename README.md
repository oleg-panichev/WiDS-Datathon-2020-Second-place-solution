# WiDS-Datathon-2020-Second-place-solution
[WiDS Datathon 2020](https://www.kaggle.com/c/widsdatathon2020) Second place solution

The code, provided here, allows to generate a final submission, that scored **0.91378** on the public leaderboard and **0.91485** on the private.
* Python 3.7
* Install Kinoa from [https://github.com/oleg-panichev/kinoa](https://github.com/oleg-panichev/kinoa) or just comment all related code

## Description
Detailed description is posted in [Kaggle discussions](https://www.kaggle.com/c/widsdatathon2020/discussion/132387).

## Running the code

Clone this repository:
```sh
git clone https://github.com/oleg-panichev/WiDS-Datathon-2020-Second-place-solution.git
```

Put the data in the 'input' directory and run the code:

```sh
cd WiDS-Datathon-2020-Second-place-solution/src
nohup python -u ensemble.py > ensemble.log &
```

You may find the code for each model within ensemble along with experiments log in the corresponding directories within \__kinoa\__ directory.

## Authors
- Iryna Ivanenko
- Oleg Panichev
