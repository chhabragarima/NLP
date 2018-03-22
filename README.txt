Language:
------------------
Programming language used - python2/3.6


File details:
------------------
1. 'infoextract.sh' is a script which installs the packages needed for the python program to run and runs the program.

2. 'infoextract.py' is the python program file for information extraction.

3. 'weapon.txt' is a file contains list of weapons which need for weapon feature to work.

4. 'individual.txt' and 'skip_words1.txt' are files that contains list of words needed for perperator individual feature to work.

5. 'target.txt' and 'skip_words2.txt' are files that contains list of words needed for target feature to work.

6. 'organisation.txt' is a file which contains list of words needed for organisation feature to work.

7. 'victim.txt' and 'names.txt' are files that contains list of words needed for victim feature to work.

8. 'stanford-corenlp-full-2017-06-09.zip' is needed to run the program. Unzipping and installing taken care inside the script. This package need to be in the same directory for the program to run.

9. 'testset1-input.txt' is a sample input file.

10. 'testset1-input.txt.templates' is a sample trace file.

11. 'developset' is a folder that has 429 articles and 429 answer keys.

Running Method
-----------------

a) For information extraction, please run the following command

	./infoextract.sh <textfile>

   Example:
	./infoextract.sh test.txt

The above command will generate the output template in the following format,

	testset1-input.txt.templates

Time Estimation:
----------------
1. Unzipping and installing the relevant packages takes few minutes.
2. For a single article it takes approximately 2 minutes. For the development set (329 files), it took approximately 30-40 minutes.


Contribution:
--------------

7 templates and modules are splitted among each other.

1. Vinu Sreenivasan - Created main modules of the system and Implemented four slots (ID, INCIDENT, TARGET,PERPETRATOR ORGANISATION)
2. Garima chhabra - Implemented three slots of the templates (WEAPON, VICTIM, PERPETRATOR INDIVIDUAL)

