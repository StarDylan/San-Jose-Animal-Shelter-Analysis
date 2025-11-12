# EDA

## Dataset

I am using the animal shelter dataset from the San Jose animal shelter, I picked this dataset
due to the proximity of the animal shelter to my hometown, and because it is quite
comprehensive. As it has records of tens of thousands of animals that have passed through the
animal shelter. It is also directly from the City of San Jose, who seems to update it
quite regularly.

The dataset includes records about lost/found reports, stray and owned animals spay/neuter
clinics, returned or confiscated animals, as well as the usual stray animals that are
turned in.

## EDA Analysis Results

Our primary goal for analysis is to predict the outcome / adoption time for cats. Therefore I filtered
out any other animals, as well as other outcomes that are more ambiguous and wouldn't contain a signal,
such as SPAY/NEUTER clinic, LOST/FOUNT animal reports, and MISSING.

The data includes intake type (e.g., stray, owner returned, return from foster family), animal condition (e.g., healthy, unhealthy, medical emergency), as well as some other factors that
may influence adoption rates such as color, breed, age.

The main signal that we want to predict is the outcome type, with the most useful types being Adoption, Euth, and Foster. There is also Rescue and Transfer, but those are less clear on the ultimate outcome.

We can also notice quite a few missing values:

```
AnimalID - 0.0%
AnimalName - 56.1%
AnimalType - 0.0%
PrimaryColor - 0.0%
SecondaryColor - 67.2%
PrimaryBreed - 0.0%
Sex - 0.0%
DOB - 32.5%
Age - 0.0%
IntakeDate - 0.0%
IntakeCondition - 0.0%
IntakeType - 0.0%
IntakeSubtype - 13.6%
IntakeReason - 96.3%
OutcomeDate - 0.0%
OutcomeType - 0.0%
OutcomeCondition - 4.3%
```

Most of these are expected such as name (e.g., for strays), secondary color (some cats are only one color), DOB (usually we don't know this for strays, also note Age is simply "NO AGE" for those without a DOB which is why it doesn't say it is missing).

One potential problem is missing outcome conditions, as pretty uniformly there seems to be about 5% of
records missing the outcome condition.

Sex is also misleading as we often don't know the sex of young kittens (some spays end up turning into neuters and vice versa) and those are marked with an "Unknown", thus appearing to not be None in our calculation.

We can also see the trend of number of cat intake over time:

![](./imgs/Intakes%20of%20Animals%20by%20Month.svg)

We can see for all the years, that May seems to be when the most cats are
being intaken by the shelter, which lines up pretty well with the kitten
season.

Another interesting piece of info is the shape of time in shelter, with most cats spending very little time
in the shelter. 50% of cats spend 4 days or less in the shelter.

![](./imgs/Time%20in%20Shelter.svg)

But when we break it out by common intakes + all outcomes, we see that cats that come from different
backgrounds spend different amoounts of time in the shelter.

![](./imgs/Time%20in%20Shelter%20by%20Outcome%20+%20Intake.svg)

We see that cats that come from a foster family spend much shorter time before adoption, most often
0 days. We also see that they much more likely to get placed in a rescue (non-profit vs. shelters which are government funded, they can also be selective about what animals they bring in) rather than transfered.
They are also almost never euthanized.

While cats that are strays are much more often euthanized, transfered, or returned to the field than foster cats. We also see that transfers take at least a few days, while rescues are much more instant.

## Open Questions

Some questions that still remain include data integrity:

- DOB vs Age -- Do they agree?
- Any cats mistakeningly labeled Male/Neutered that are also pregnant?

As well as the semantics of some of the codes:

- EUTH Req? Do they always result in EUTH? / Same with DISPO REQ

I also need to work out what codes are sufficiently similar for color, outcome, and condition. Because currently there are a lot of codes that seem to be redundant or just not useful.
