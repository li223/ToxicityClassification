# ToxicityClassification
Basic application for testing toxicity. Based upon the Sentiment Analysis Example in ML.Net.

# This looks same as the example
It is. Almost line by line because I have no idea how to use ML.Net and want to put a twist on the example app to see if I could learn how anything works

# I want to build it for some reason
First create 2 folders in the bottom of the folder tree (IE: netcoreapp3.0). Create a folder called Data and within this folder add your test data file in called `TestData.tsv`. Next create another folder called Model. This is where your models will go.

Format for the TestData is as follows:
0/1 "TEXT"

The 0/1 represents false/true to whether or not the following text is considered toxic. Then seperated by a TAB and in quotation marks is the toxic text.
