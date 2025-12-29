# animals-detection

Download finetuned YOLO11 model via [link](https://drive.google.com/drive/folders/1Ni5kEC2357K1jdCaID2eqEYlVQwPtsIT?usp=sharing)

1. Create venv (recommended python version is 3.10):
   ```sh
   python -m venv venv
   ```
2. Activate environment:

   ```sh
   # Windows
   ./venv/Scripts/activate

   # Linux\Mac
   source venv/bin/activate
   ```

3. Install requirements:
   ```sh
   pip install -r ./requirements.txt
   ```
4. To evaluate model and get table with ROC AUC metric, ensure that your `.yaml` file looks like:
  ```
  names:
    0: mountain_hare
    1: badger
  val: val_split.txt
  ```
  Then launch next script:
  ```
  python .\animals_detect\evaluate_model.py --weights path_to_model.pt --data path_to_data.yaml
  ```

As result, you will get excel table 'evaluation.xlsx` in such format:

| Class    | Encounters | Total | Max/Image | ROC AUC |
| -------- | -----------| ------| --------- | ------- |
| Animal 1 | 178        | 181   | 2         | 0.986   |
| Animal 2 | 15         | 15    | 1         | 1.00    |
| Animal 3 | 0          | 0     | 0         | nan     |

