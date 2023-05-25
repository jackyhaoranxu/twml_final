python main.py --data EMODB-EMOVO_RAVDE-SAVEE --train_datasets EMODB EMOVO --test_datasets RAVDE SAVEE --data_type mfcc
python main.py --data RAVDE-SAVEE_EMODB-EMOVO --train_datasets RAVDE SAVEE --test_datasets EMODB EMOVO --data_type mfcc
python main.py --data EMODB-EMOVO-RAVDE_SAVEE --train_datasets EMODB EMOVO RAVDE --test_datasets SAVEE --data_type mfcc
python main.py --data EMODB-EMOVO-SAVEE_RAVDE --train_datasets EMODB EMOVO SAVEE --test_datasets RAVDE --data_type mfcc
python main.py --data EMODB-RAVDE-SAVEE_EMOVO --train_datasets EMODB RAVDE SAVEE --test_datasets EMOVO --data_type mfcc
python main.py --data EMOVO-RAVDE-SAVEE_EMODB --train_datasets EMOVO RAVDE SAVEE --test_datasets EMODB --data_type mfcc