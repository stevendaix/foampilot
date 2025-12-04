from build123d import *
from typing import Optional, Tuple, List

class HumanGeometry:
    """
    Générateur paramétrique de géométrie humaine simplifiée pour CFD.
    Permet de définir la taille totale ('height') et la posture.
    Postures supportées : 'standing', 'seated', ou 'arms_raised'.
    """

    def __init__(
        self,
        height: Optional[float] = None,  # NOUVEAU: Hauteur totale désirée (ex: 1.80m)
        scale: float = 1.0,              # Échelle si 'height' n'est pas utilisé
        shoulder_width: float = 0.45,
        hip_width: float = 0.32,
        torso_height: float = 0.65,
        neck_height: float = 0.10,
        head_radius: float = 0.14,
        arm_radius: float = 0.07,
        leg_radius: float = 0.09,
        arm_length: float = 0.65,
        leg_length: float = 0.95,
        foot_length: float = 0.25,
        foot_height: float = 0.05,
        hand_size: float = 0.10,
        posture: str = "standing",  # 'standing', 'seated' ou 'arms_raised'
    ):
        
        # --- 1. Calcul de l'échelle ---
        if height is not None:
            # Hauteur de référence (H_ref) pour scale=1.0 (approximative)
            # Somme des longueurs des parties en Z (Leg + Torso + Neck + 2*Head_radius*1.2 + Foot_height)
            H_ref_default = 0.95 + 0.65 + 0.10 + (0.14 * 1.2 * 2) + 0.05
            
            if H_ref_default == 0: H_ref_default = 1.0 # Sécurité
            
            self.scale = height / H_ref_default
        else:
            self.scale = scale

        # --- 2. Application de l'échelle à TOUS les paramètres ---
        self.shoulder_width = shoulder_width * self.scale
        self.hip_width = hip_width * self.scale
        self.torso_height = torso_height * self.scale
        self.neck_height = neck_height * self.scale
        self.head_radius = head_radius * self.scale
        self.arm_radius = arm_radius * self.scale
        self.leg_radius = leg_radius * self.scale
        self.arm_length = arm_length * self.scale
        self.leg_length = leg_length * self.scale
        self.foot_length = foot_length * self.scale
        self.foot_height = foot_height * self.scale
        self.hand_size = hand_size * self.scale
        self.posture = posture.lower()

        # Cache de la géométrie
        self._human_part: Optional[Part] = None

# ----------------- Construction des parties -----------------

    def _build_head(self) -> Part:
        with BuildPart() as head:
            Ellipsoid(
                rx=self.head_radius,
                ry=self.head_radius * 0.9,
                rz=self.head_radius * 1.2,
            )
        return head.part

    def _build_neck(self) -> Part:
        with BuildPart() as neck:
            Cylinder(radius=self.head_radius * 0.45, height=self.neck_height)
        return neck.part

    def _build_torso(self) -> Part:
        with BuildPart() as torso:
            with BuildSketch() as s1:  # Hanches
                Rectangle(self.hip_width, self.hip_width * 0.45)
            with BuildSketch() as s2:  # Poitrine
                Rectangle(self.shoulder_width * 0.9, self.hip_width * 0.6)
                s2.local_loc = Location((0, 0, self.torso_height * 0.5))
            with BuildSketch() as s3:  # Épaules
                Rectangle(self.shoulder_width, self.shoulder_width * 0.35)
                s3.local_loc = Location((0, 0, self.torso_height))
            Loft([s1.sketch, s2.sketch, s3.sketch], loft_type=LoftType.SMOOTH)
        return torso.part

    def _build_limb(self, radius: float, length: float, rotation=(0, 0, 0)) -> Part:
        """Cylindre pour bras ou jambe, avec rotation optionnelle."""
        with BuildPart() as limb:
            Cylinder(radius=radius, height=length, rotation=rotation)
        return limb.part
        
    def _build_foot(self) -> Part:
        """Crée un pied rectangulaire (Box)."""
        with BuildPart() as foot:
            Box(
                self.foot_length,                 # Longueur (X)
                self.leg_radius * 1.5,            # Largeur (Y)
                self.foot_height,                 # Hauteur (Z)
                centered=(True, True, False) # Base du pied à Z=0
            )
        return foot.part

    def _build_hand(self) -> Part:
        """Crée une main cubique simplifiée (Box)."""
        with BuildPart() as hand:
            Box(self.hand_size, self.hand_size, self.hand_size)
        return hand.part


# ----------------- Assemblage postures -----------------

    def _assemble_standing(self, torso: Part, neck: Part, head: Part) -> Part:
        parts: List[Part] = [torso]

        # Tête et cou
        parts.append(neck.located(Location((0, 0, self.torso_height))))
        parts.append(head.located(Location((0, 0, self.torso_height + self.neck_height + self.head_radius * 1.2))))

        # Jambes et pieds
        leg = self._build_limb(self.leg_radius, self.leg_length)
        foot = self._build_foot()
        hip_offset = self.hip_width * 0.33
        
        # Le bas de la jambe est à z = -self.leg_length
        leg_z_center = -self.leg_length / 2
        leg_z_bottom = -self.leg_length
        foot_z_center = leg_z_bottom - self.foot_height / 2 

        for sign in [-1, 1]:
            # Jambe
            parts.append(leg.located(Location((sign * hip_offset, 0, leg_z_center))))
            # Pied (positionné sous la jambe, longueur orientée sur Y)
            parts.append(foot.located(Location((sign * hip_offset, self.foot_length / 2, foot_z_center))))

        # Bras et mains
        arm = self._build_limb(self.arm_radius, self.arm_length)
        hand = self._build_hand()
        arm_z_center = self.torso_height * 0.5 - self.arm_length / 2
        arm_x = self.shoulder_width * 0.6
        
        arm_z_bottom = arm_z_center - self.arm_length / 2
        hand_z_center = arm_z_bottom - self.hand_size / 2

        for sign in [-1, 1]:
            # Bras
            parts.append(arm.located(Location((sign * arm_x, 0, arm_z_center))))
            # Main
            parts.append(hand.located(Location((sign * arm_x, 0, hand_z_center))))

        return Compound.make_compound(parts)

    def _assemble_seated(self, torso: Part, neck: Part, head: Part) -> Part:
        parts: List[Part] = [torso]

        # Tête et cou
        parts.append(neck.located(Location((0, 0, self.torso_height))))
        parts.append(head.located(Location((0, 0, self.torso_height + self.neck_height + self.head_radius * 1.2))))

        # Jambes pliées 90° et pieds
        thigh_len = self.leg_length * 0.5
        shin_len = self.leg_length * 0.5
        hip_offset = self.hip_width * 0.33
        foot = self._build_foot()

        shin_z_bottom = -shin_len
        foot_z_center = shin_z_bottom - self.foot_height / 2

        for sign in [-1, 1]:
            # Cuisse (horizontale, rotation 90 autour de X pour être sur Y)
            thigh = self._build_limb(self.leg_radius, thigh_len, rotation=(90, 0, 0))
            thigh_loc = Location((sign * hip_offset, thigh_len / 2, 0))
            parts.append(thigh.located(thigh_loc))

            # Tibia (vertical)
            shin = self._build_limb(self.leg_radius, shin_len)
            shin_loc = Location((sign * hip_offset, thigh_len, -shin_len / 2))
            parts.append(shin.located(shin_loc))
            
            # Pied (positionné au bout du tibia)
            foot_loc = Location((sign * hip_offset, thigh_len + self.foot_length / 2, foot_z_center))
            parts.append(foot.located(foot_loc))

        # Bras et mains (verticaux, comme standing)
        arm = self._build_limb(self.arm_radius, self.arm_length)
        hand = self._build_hand()
        arm_z_center = self.torso_height * 0.5 - self.arm_length / 2
        arm_x = self.shoulder_width * 0.6
        
        arm_z_bottom = arm_z_center - self.arm_length / 2
        hand_z_center = arm_z_bottom - self.hand_size / 2

        for sign in [-1, 1]:
            parts.append(arm.located(Location((sign * arm_x, 0, arm_z_center))))
            parts.append(hand.located(Location((sign * arm_x, 0, hand_z_center))))

        return Compound.make_compound(parts)

    def _assemble_arms_raised(self, torso: Part, neck: Part, head: Part) -> Part:
        """Assemble le corps en posture 'bras levés'."""
        parts: List[Part] = [torso]

        # Tête et cou (IDEM standing)
        parts.append(neck.located(Location((0, 0, self.torso_height))))
        parts.append(head.located(Location((0, 0, self.torso_height + self.neck_height + self.head_radius * 1.2))))

        # Jambes et pieds (IDEM standing)
        leg = self._build_limb(self.leg_radius, self.leg_length)
        foot = self._build_foot()
        hip_offset = self.hip_width * 0.33
        leg_z_center = -self.leg_length / 2
        leg_z_bottom = -self.leg_length
        foot_z_center = leg_z_bottom - self.foot_height / 2 

        for sign in [-1, 1]:
            parts.append(leg.located(Location((sign * hip_offset, 0, leg_z_center))))
            parts.append(foot.located(Location((sign * hip_offset, self.foot_length / 2, foot_z_center))))

        # Bras levés (vertical le long de Z, partant de l'épaule)
        arm = self._build_limb(self.arm_radius, self.arm_length) 
        arm_len = self.arm_length
        hand = self._build_hand()
        arm_x = self.shoulder_width * 0.6
        
        # Le bas du bras est à z = torso_height. 
        arm_z_center = self.torso_height + arm_len / 2
        
        # Haut du bras (main)
        arm_z_top = self.torso_height + arm_len
        hand_z_center = arm_z_top + self.hand_size / 2

        for sign in [-1, 1]:
            # Bras levé
            arm_loc = Location((sign * arm_x, 0, arm_z_center))
            parts.append(arm.located(arm_loc))
            
            # Main
            hand_loc = Location((sign * arm_x, 0, hand_z_center))
            parts.append(hand.located(hand_loc))

        return Compound.make_compound(parts)


# ----------------- Interface publique -----------------

    def build_human(self) -> Part:
        """Construit le corps humain selon la posture."""
        if self._human_part is not None:
            return self._human_part

        torso, neck, head = self._build_torso(), self._build_neck(), self._build_head()

        if self.posture == "standing":
            self._human_part = self._assemble_standing(torso, neck, head)
        elif self.posture == "seated":
            self._human_part = self._assemble_seated(torso, neck, head)
        elif self.posture == "arms_raised":
            self._human_part = self._assemble_arms_raised(torso, neck, head)
        else:
            print(f"Posture '{self.posture}' non reconnue. Utilisation de 'standing'.")
            self._human_part = self._assemble_standing(torso, neck, head)

        return self._human_part

    def build_cfd_domain(self, domain_size: Tuple[float, float, float], subtract_human: bool = True) -> Part:
        """Crée un domaine CFD (boîte) et soustrait le corps humain si demandé."""
        human_part = self.build_human()
        Lx, Ly, Lz = domain_size
        with BuildPart() as cfd_box:
            # Centre la boîte sur le corps humain
            Box(Lx, Ly, Lz, centered=(True, True, False)).move(Location((0, 0, Lz/2)))
        cfd_domain = cfd_box.part

        if subtract_human:
            # Nécessite de fusionner le corps humain en un seul solide pour la soustraction
            fused_human = Compound.make_compound(human_part.solids()).fuse()
            cfd_domain = cfd_domain - fused_human

        return cfd_domain

    def export_step_all(self, human_filename: str = "human_model.step", domain_filename: Optional[str] = None,
                        domain_size: Optional[Tuple[float, float, float]] = None) -> List[str]:
        exported_files: List[str] = []

        # Corps humain seul
        self.build_human().export_step(human_filename)
        exported_files.append(human_filename)

        # Domaine CFD optionnel
        if domain_filename and domain_size:
            domain_part = self.build_cfd_domain(domain_size=domain_size, subtract_human=True)
            domain_part.export_step(domain_filename)
            exported_files.append(domain_filename)
        elif domain_filename and not domain_size:
            print("AVERTISSEMENT: 'domain_filename' spécifié mais 'domain_size' manquant. Export du domaine omis.")

        return exported_files
