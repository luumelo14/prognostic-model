# prognostic-model
Prognostic model for traumatic brachial plexus injuries 

1. Open a terminal and clone this project 
```
$ git clone https://github.com/luumelo14/prognostic-model.git
```

2. Create a virtual environment and activate it 

  - On Linux/MacOS:
    ```
    $ python3 -m venv env
    $ source env/bin/activate
    ```

  - On Windows:
    ```
    $ py -m venv env
    $ env\Scripts\activate
    ```
    
3. Install the project's dependencies:
```
$ (env) pip install -r requirements.txt
```

4. Navigate to [Brachial Plexus Injury Database](https://neuromatdb.numec.prp.usp.br/experiments/brachial-plexus-injury-database/), click on Downloads tab and then "Download all experiment data"

5. Extract the files to a directory of your preference. 

6. Run preprocess.py and follow the menu prompts accordingly to preprocess the files:
```
$ (env) python preprocess.py
```

7. Run comparison.py to train the models:
```
$ (env) python comparison.py
```
