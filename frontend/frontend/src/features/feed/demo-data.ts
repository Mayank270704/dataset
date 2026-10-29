import type { PolicyPost } from "./types";

/**
 * Five recent Ministry of Home Affairs policy, programme, and reform items.
 * Each sourceUrl points to an official MHA/PIB release. These are informational
 * records, not claims that the Government has opened a public consultation.
 * openedAt controls only PolicyLens's independent four-day preview window.
 */
export const officialPolicies: PolicyPost[] = [
  {
    id: "mha-prahaar-ct-policy",
    slug: "prahaar-counter-terrorism-policy",
    title: "PRAHAAR — National Counter-Terrorism Policy & Strategy",
    summary: "The Ministry of Home Affairs describes PRAHAAR as a national counter-terrorism framework covering prevention, response, coordinated capacity, human-rights and rule-of-law safeguards, international cooperation, and recovery and resilience.",
    category: "Public services",
    ministry: "Ministry of Home Affairs",
    publishedDate: "28 Sep 2026",
    sourceName: "Press Information Bureau (PIB) — Ministry of Home Affairs",
    sourceUrl: "https://www.pib.gov.in/PressReleasePage.aspx?PRID=2316153",
    statusLabel: "National Security Strategy",
    verifiedOfficial: true,
    language: "English / Hindi",
    department: "Internal Security",
    schemeType: "National policy and strategy",
    openedAt: "2026-10-14T00:00:00+05:30",
    consultationStatus: "open",
    keyHighlights: [
      "The policy is organized around seven pillars, including prevention, response, internal coordination, human rights, international cooperation, and recovery and resilience.",
      "The 28 September release reports the launch of an Extradition Portal to coordinate efforts to bring fugitives back to India.",
      "This is an official policy information item; the PolicyLens feedback window is independent of the Government."
    ]
  },
  {
    id: "mha-special-campaign-6",
    slug: "special-campaign-6-mha",
    title: "Special Campaign 6 — Grievances and Record Management",
    summary: "The Ministry of Home Affairs is participating in Special Campaign 6, with a focus on cleaner and more sustainable workplaces, improved record management, and timely disposal of public grievances and pending references.",
    category: "Public services",
    ministry: "Ministry of Home Affairs",
    publishedDate: "23 Sep 2026",
    sourceName: "Press Information Bureau (PIB) — Ministry of Home Affairs",
    sourceUrl: "https://www.pib.gov.in/PressReleasePage.aspx?PRID=2313801",
    statusLabel: "Public Administration",
    verifiedOfficial: true,
    language: "English / Hindi",
    department: "Administration",
    schemeType: "Government campaign",
    openedAt: "2026-10-14T00:00:00+05:30",
    consultationStatus: "open",
    keyHighlights: [
      "The release lists the implementation phase as 2–31 October 2026.",
      "Focus areas include pending public grievances and appeals, government references, and improved record management.",
      "The ministry says field and outstation offices that interface directly with the public receive special attention."
    ]
  },
  {
    id: "mha-national-drug-control-vision",
    slug: "national-drug-control-vision-2026-2029",
    title: "National Drug Control Vision (2026–2029)",
    summary: "The Ministry of Home Affairs' national drug-control vision sets out a three-year roadmap to coordinate action against illicit drug trafficking and abuse across enforcement, precursor and synthetic-drug control, demand and harm reduction, and capacity building.",
    category: "Public services",
    ministry: "Ministry of Home Affairs",
    publishedDate: "28 Jul 2026",
    sourceName: "Press Information Bureau (PIB) — Ministry of Home Affairs",
    sourceUrl: "https://www.pib.gov.in/PressReleasePage.aspx?PRID=2290379",
    statusLabel: "National Action Roadmap",
    verifiedOfficial: true,
    language: "English / Hindi",
    department: "Narcotics Control",
    schemeType: "National vision document",
    openedAt: "2026-10-14T00:00:00+05:30",
    consultationStatus: "open",
    keyHighlights: [
      "The three-year vision document is for 2026–2029.",
      "Four focus areas cover enforcement, precursor and synthetic drugs control, demand and harm reduction, and capacity building and coordination.",
      "The source describes a government action roadmap; the PolicyLens commenting period is separate and independent."
    ]
  },
  {
    id: "mha-ai-cyber-crime",
    slug: "ai-enabled-cyber-crime-prevention",
    title: "AI-Enabled Cybercrime Prevention and Digital Identity Protection",
    summary: "The Ministry of Home Affairs outlines measures to strengthen coordinated responses to cybercrime, including AI-enabled offences, and support cyber resilience through policy guidance, technology, capacity building, and cooperation with State and Union Territory agencies.",
    category: "Digital government",
    ministry: "Ministry of Home Affairs",
    publishedDate: "12 Aug 2026",
    sourceName: "Press Information Bureau (PIB) — Ministry of Home Affairs",
    sourceUrl: "https://www.pib.gov.in/PressReleasePage.aspx?PRID=2298336",
    statusLabel: "Cybercrime Prevention",
    verifiedOfficial: true,
    language: "English / Hindi",
    department: "Indian Cyber Crime Coordination Centre (I4C)",
    schemeType: "Cybercrime prevention measures",
    openedAt: "2026-10-14T00:00:00+05:30",
    consultationStatus: "open",
    keyHighlights: [
      "The release addresses coordinated prevention and investigation of cybercrime, including AI-enabled cybercrime.",
      "MHA describes support for States and Union Territories through policy guidance, technology, training, and financial resources.",
      "The Indian Cyber Crime Coordination Centre (I4C) is part of the described national response."
    ]
  },
  {
    id: "mha-immigration-ivfrt",
    slug: "immigration-visa-foreigners-registration-tracking",
    title: "Immigration, Visa, Foreigners Registration & Tracking (IVFRT)",
    summary: "The Union Cabinet approved continuation of the IVFRT scheme for five years, from 1 April 2026 through 31 March 2031. The platform is intended to connect immigration, visa issuance, and foreigners' registration and tracking services.",
    category: "Digital government",
    ministry: "Ministry of Home Affairs",
    publishedDate: "25 Mar 2026",
    sourceName: "Press Information Bureau (PIB) — Ministry of Home Affairs",
    sourceUrl: "https://www.pib.gov.in/PressReleasePage.aspx?PRID=2245101",
    statusLabel: "Immigration Services",
    verifiedOfficial: true,
    language: "English / Hindi",
    department: "Immigration",
    schemeType: "Digital public service scheme",
    openedAt: "2026-10-14T00:00:00+05:30",
    consultationStatus: "open",
    keyHighlights: [
      "The approved continuation period runs from 1 April 2026 to 31 March 2031.",
      "The PIB release states a budget outlay of ₹1,800 crore.",
      "The platform is intended to modernize integrated immigration and visa service delivery while supporting national security."
    ]
  }
];

/** Backwards compatibility for older frontend imports. */
export const demoPosts = officialPolicies;
