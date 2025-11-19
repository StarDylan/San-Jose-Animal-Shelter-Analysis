# San Jose Animal Shelter Analysis

## Overview

This project analyzes animal intake and outcome data from the San Jose Animal Shelter, focusing specifically on cats. The goal is to predict the outcome / adoption time for cats.

## Data

Dataset Website: https://data.sanjoseca.gov/dataset/animal-shelter-intake-and-outcomes

This dataset is quite comprehensive. It has records of tens of thousands of
animals that have passed through the animal shelter. The dataset includes records about
lost/found reports, stray and owned animals spay/neuter clinics, returned or confiscated animals,
as well as the usual stray animals that are turned in.

Key features:

- **Animal Characteristics**: Animal Type, Color, Breed, Sex, Age
- **Shelter Events**: Intake + Outcome Date, Condition, and Type

## Report Draft:

We used a random forest as a baseline to predict whether a given cat would be adopted or not. We used these features in building the baseline model: "IntakeMonth",
"IntakeMedicalIssueIndex",
"IntakeType",
"Sex",
"PrimaryBreed",
"PrimaryColor",
"SecondaryColor",
"IntakeIsNursing",
"SpayedNeutered".

On binary classification. We achieved a precision of 0.75, recall of 0.74 and an f1 score of 0.75. Below is the plot of permutation importance for each of
the features:

![](./imgs/rf_permutation_importance.png)

We see that Age days has a very large importance (which makes sense, as older cats will be eligible for adoption while very young cats (a large proportion of intaked cats are often placed into foster care)). But we can also see a large impact of SpayedNeutered, which could potentially be indiciative of target leakage, as maybe cats who are being adopeted will be spayed/neutered beforehand, while other cats won't be.

For future work, we will further investigate the relationship between adoptions and spayed/neutered, as well as the impact of age and whether there are any other features
that could be useful in determining adoption status. We can also regress on Time in Shelter for adopted cats, so we can estimate how long a cat might stay at the shelter.
