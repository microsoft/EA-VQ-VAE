# Copyright (c) Microsoft Corporation. 
# Licensed under the MIT license.

wget https://uwnlp.github.io/event2mind/data/event2mind.zip
mkdir -p event2mind
mv event2mind.zip event2mind

unzip event2mind/event2mind.zip -d event2mind
rm event2mind/event2mind.zip

mv event2mind/train.csv event2mind/trn.csv
mv event2mind/test.csv event2mind/tst.csv
